import json
import torch
import vllm
from vllm.attention import Attention
from vllm.model_executor.models.llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
)
from typing import Optional, List, Tuple
from vllm import _custom_ops as vllm_ops
from vllm.attention.ops.paged_attn import PagedAttention
from vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
import torch.nn.functional as F


def quest_patch_vllm(
    llm,
    args
):
    vllm_version = vllm.__version__

    def update_module(m):
        if isinstance(m, Attention):
            m.forward = vllm_attn_forward.__get__(m, Attention)

            m = m.impl
            m_cls = m.__class__
            m.args = args

            m.forward = quest_vllm_forward.__get__(m, m_cls)
        if isinstance(m, LlamaDecoderLayer):
            m.forward = llama_layer_forward_vllm.__get__(m, LlamaDecoderLayer)
        if isinstance(m, LlamaModel):
            m.forward = llama_model_forward_vllm.__get__(m, LlamaModel)
        if isinstance(m, LlamaAttention):
            m.forward = llama_attn_forward_vllm(vllm_version).__get__(m, LlamaAttention)

    llm.llm_engine.model_executor.driver_worker.model_runner.model.apply(update_module)

    print("Patched model for quest with vLLM..")
    return llm


def llama_model_forward_vllm(
    self,
    input_ids: Optional[torch.Tensor],
    positions: torch.Tensor,
    kv_caches: List[torch.Tensor],
    attn_metadata,
    inputs_embeds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if inputs_embeds is not None:
        hidden_states = inputs_embeds
    else:
        hidden_states = self.get_input_embeddings(input_ids)
    residual = None
    for i in range(len(self.layers)):
        layer = self.layers[i]
        hidden_states, residual = layer(
            positions,
            hidden_states,
            kv_caches[i],
            attn_metadata,
            residual,
            layer_idx=i,
        )
    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states


def llama_layer_forward_vllm(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    residual: Optional[torch.Tensor],
    layer_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Self Attention
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    hidden_states = self.self_attn(
        positions=positions,
        hidden_states=hidden_states,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        layer_idx=layer_idx,
    )

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual


def llama_attn_forward_vllm(
    vllm_version: str = "0.4.2",
):
    def llama_attn_forward_vllm(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        layer_idx: int,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        if "0.4.1" <= vllm_version <= "0.4.2":
            attn_output = self.attn(
                q, k, v, kv_cache, attn_metadata, self.kv_scale, layer_idx
            )
        elif vllm_version >= "0.4.3":
            attn_output = self.attn(q, k, v, kv_cache, attn_metadata, layer_idx)
        else:
            assert False, "Only support 'vllm>=0.4.1'. Please update your vllm version."

        output, _ = self.o_proj(attn_output)
        return output

    return llama_attn_forward_vllm


def vllm_attn_forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: Optional[torch.Tensor],
    attn_metadata,
    # kv_scale: float = 1.0,
    layer_idx: int = 0,
) -> torch.Tensor:
    # check self._kv_scale
    # kv_scale = getattr(self, "_kv_scale", kv_scale)
    return self.impl.forward(
        query, key, value, kv_cache, attn_metadata, self._k_scale, self._v_scale, layer_idx
    )

def quest_vllm_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        layer_idx: int = 0,
    ) -> torch.Tensor:
    """Forward pass with FlashAttention.

    Args:
        query: shape = [num_tokens, num_heads * head_size]
        key: shape = [num_tokens, num_kv_heads * head_size]
        value: shape = [num_tokens, num_kv_heads * head_size]
        kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
        attn_metadata: Metadata for attention.
    Returns:
        shape = [num_tokens, num_heads * head_size]
    """

    assert k_scale == 1.0 and v_scale == 1.0, (
        "key/v_scale is not supported in FlashAttention.")

    def repeat_kv(hidden_states, n_rep):
        sqlen, chunk_size, num_head, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:,:, :, None, :].expand(sqlen, chunk_size, num_head, n_rep, head_dim)
        return hidden_states.reshape(sqlen, chunk_size, num_head * n_rep, head_dim)

    def sparse_attention_fixed_mask(q, k, mask, v):
        B, H, Q, D = q.size()
        assert Q == 1
        _, _, K, D_v = v.size()

        q = q.view(B * H, Q, D)         # (B*H, Q, D)
        k = k.view(B * H, K, D)         # (B*H, K, D)
        v = v.view(B * H, K, D_v)       # (B*H, K, D_v)
        mask = mask.view(B * H, Q, K)   # (B*H, Q, K)

        mask = mask.squeeze(1)           # (B*H, K)  

        mask_counts = mask.float().sum(dim=1)
        assert torch.all(mask_counts == mask_counts[0]), "不同头部中 mask 为 True 的数量不相同"
        M = int(mask_counts[0].item())  # 每个头部中 mask 为 True 的数量

        indices = mask.float().topk(M, dim=1, largest=True, sorted=False).indices  # (B*H, M)

        batch_indices = torch.arange(B * H).unsqueeze(1).to(q.device).expand(-1, M)  # (B*H, M)

        k_masked = k[batch_indices, indices]  # (B*H, M, D)
        v_masked = v[batch_indices, indices]  # (B*H, M, D_v)

        scores = torch.bmm(q.float(), k_masked.transpose(1, 2).float())  # (B*H, Q, M)

        scaling_factor = torch.sqrt(torch.tensor(D, dtype=torch.float32, device=q.device))
        scores = scores / scaling_factor

        attn_weights = torch.softmax(scores, dim=-1)  # (B*H, Q, M)
 
        output = torch.bmm(attn_weights, v_masked.float())   # (B*H, Q, D_v)

        output = output.view(B, H, Q, D_v)            # (B, H, Q, D_v)
        output = output.squeeze(2)

        return output

    def decode_quest_func(query, key_cache, value_cache, block_table,seq_lens):
        page_indices = block_table.squeeze(0)
        key = key_cache[page_indices] # (8146,16,8,128)
        value = value_cache[page_indices]
        key = repeat_kv(key, query.shape[-2]//key.shape[-2]) # (8146,16,32,128)
        value = repeat_kv(value, query.shape[-2]//value.shape[-2])
        chunk_size, num_heads = key.shape[1], key.shape[2]
        padding_length = chunk_size - seq_lens % chunk_size
        if query.dtype == torch.bfloat16:
            min_value = -1e20
            max_value = 1e20
        elif query.dtype == torch.float16:
            min_value = -1e4
            max_value = 1e4
        else:
            print(query.dtype)
            min_value = -1e4
            max_value = 1e4
        padding_key = torch.where(
            query > 0,
            torch.full_like(query, min_value),
            torch.full_like(query, max_value)
        ).to(key.device)
        fill_keys = padding_key.expand(padding_length, -1, -1)
        key[-1,-padding_length:,:,:] = fill_keys
        query = query.unsqueeze(2) # (1,32,1,128)
        key = key.permute(2,0,1,3).unsqueeze(0) # (1,32,8146,16,128)
        value = value.permute(2,0,1,3).unsqueeze(0)
        key_2 = key.reshape(key.shape[0],key.shape[1],key.shape[2]*key.shape[3],key.shape[4])
        value_2 = value.reshape(value.shape[0],value.shape[1],value.shape[2]*value.shape[3],value.shape[4])

        device = key.device
        total_size = key_2.shape[-2]

        if self.args.token_budget >= seq_lens:
                quest_need_estimate = False
        else:
            quest_need_estimate = True

        if self.args.quest == True and quest_need_estimate == True:
            bsz = 1
            key_min, _ = key.min(dim=3)  # 形状为 (1, 32, num_chunks, 128)
            key_max, _ = key.max(dim=3)
            mul_min = key_min * query  
            mul_max = key_max * query 
            max_mul = torch.max(mul_min, mul_max)  
            chunk_score = max_mul.sum(dim=-1)
            _, topk_indices = torch.topk(chunk_score,self.args.token_budget // chunk_size,dim=-1)

            range_vec = torch.arange(chunk_size, device=device).view(1, 1, 1, chunk_size)
            topk_indices_expanded = topk_indices.unsqueeze(-1)
            expanded_indices = topk_indices_expanded * chunk_size + range_vec
            expanded_indices = expanded_indices.view(bsz, num_heads, -1)
            mask = torch.zeros(bsz, num_heads, 1, total_size, device=device, dtype=torch.float32)
            scatter_indices = expanded_indices.unsqueeze(2)
            src = torch.ones_like(scatter_indices, dtype=mask.dtype)
            mask.scatter_(dim=-1, index=scatter_indices, src=src)
        else:
            mask = torch.ones((1, num_heads, 1, total_size), device=device, dtype=torch.float32)
        attn_output = sparse_attention_fixed_mask(query,key_2,mask,value_2).to(query.dtype)
        
        return attn_output

    num_tokens, hidden_size = query.shape
    # Reshape the query, key, and value tensors.
    query = query.view(-1, self.num_heads, self.head_size)
    key = key.view(-1, self.num_kv_heads, self.head_size)
    value = value.view(-1, self.num_kv_heads, self.head_size)

    if kv_cache is not None:
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]

        vllm_ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            attn_metadata.slot_mapping.flatten(),
            self.kv_cache_dtype,
            k_scale,
            v_scale,
        )

    num_prefill_tokens = attn_metadata.num_prefill_tokens # 512
    num_decode_tokens = attn_metadata.num_decode_tokens # 0
    assert key.shape[0] == num_prefill_tokens + num_decode_tokens
    assert value.shape[0] == num_prefill_tokens + num_decode_tokens

    output = torch.empty_like(query)
    # Query for decode. KV is not needed because it is already cached.
    decode_query = query[num_prefill_tokens:]
    # QKV for prefill.
    query = query[:num_prefill_tokens]
    key = key[:num_prefill_tokens]
    value = value[:num_prefill_tokens]

    assert query.shape[0] == num_prefill_tokens
    assert decode_query.shape[0] == num_decode_tokens

    prefill_output: Optional[torch.Tensor] = None
    decode_output: Optional[torch.Tensor] = None

    if prefill_meta := attn_metadata.prefill_metadata:
        # Prompt run.
        if (kv_cache is None or prefill_meta.block_tables is None
                or prefill_meta.block_tables.numel() == 0):
            # normal attention
            # When block_tables are not filled, it means q and k are the
            # prompt, and they have the same length.
            prefill_output = flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=prefill_meta.seq_start_loc,
                cu_seqlens_k=prefill_meta.seq_start_loc,
                max_seqlen_q=prefill_meta.max_prefill_seq_len,
                max_seqlen_k=prefill_meta.max_prefill_seq_len,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.sliding_window,
                alibi_slopes=self.alibi_slopes,
                softcap=self.logits_soft_cap,
            )
        else:
            # prefix-enabled attention
            assert prefill_meta.seq_lens is not None
            max_seq_len = max(prefill_meta.seq_lens)
            prefill_output = flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=prefill_meta.query_start_loc,
                max_seqlen_q=prefill_meta.max_query_len,
                cu_seqlens_k=prefill_meta.seq_start_loc,
                max_seqlen_k=max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                block_table=prefill_meta.block_tables,
                softcap=self.logits_soft_cap,
            )
    if decode_meta := attn_metadata.decode_metadata:
        # Decoding run.
        if layer_idx < 2:
            decode_output = flash_attn_with_kvcache(
                decode_query.unsqueeze(1),
                key_cache,
                value_cache, 
                block_table=decode_meta.block_tables,
                cache_seqlens=decode_meta.seq_lens_tensor,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                softcap=self.logits_soft_cap,
            ).squeeze(1)
        else:
            decode_output = decode_quest_func(decode_query, key_cache, value_cache, decode_meta.block_tables, decode_meta.max_decode_seq_len)

    if prefill_output is None:
        assert decode_output is not None
        return decode_output.view(num_decode_tokens, hidden_size)
    if decode_output is None:
        assert prefill_output is not None
        return prefill_output.view(num_prefill_tokens, hidden_size)
    output = torch.cat([prefill_output, decode_output], dim=0)

    return output.view(num_tokens, hidden_size)
