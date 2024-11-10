import math
import numpy as np
from typing import Optional, Tuple, Union
import os
import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
import types

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    apply_rotary_pos_emb,
    repeat_kv,
)

def local_heavy_hitter_mask(attn_weights, token_budget, chunk_size):
    # attn_weights (BS, head, query, keys)

    # expend attn_weights to be divisible by chunk_size
    seq_length = attn_weights.shape[-1]
    padding_length = chunk_size - ((seq_length - 1) % chunk_size + 1)
    attn_weights = torch.cat(
        [
            attn_weights,
            torch.ones(
                (
                    attn_weights.shape[0],
                    attn_weights.shape[1],
                    attn_weights.shape[2],
                    padding_length,
                ),
                device=attn_weights.device,
            )
            * torch.tensor(torch.finfo(attn_weights.dtype).min),
        ],
        dim=-1,
    )

    # chunk attn_weights into chunk_size tokens
    chunk_attn_weights = attn_weights.reshape(
        attn_weights.shape[0],
        attn_weights.shape[1],
        attn_weights.shape[2],
        attn_weights.shape[3] // chunk_size,
        chunk_size,
    ).amax(dim=-1)

    _, topk = chunk_attn_weights.topk(
        k=min(max(3, token_budget // chunk_size), chunk_attn_weights.size(-1)), dim=-1
    )
    # repeat topk chunk_size times and recover the original indexes (* chunk_size + arange(chunk_size))
    topk = topk.unsqueeze(-1).repeat(
        1, 1, 1, 1, chunk_size
    ) * chunk_size + torch.arange(chunk_size, device=topk.device)
    topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)
    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom.scatter_(-1, topk, True)

    # remove the padding
    mask_bottom = mask_bottom[:, :, :, :seq_length]

    return mask_bottom

global layer_id
layer_id = 32

class CustomLlamaAttention(LlamaAttention):
    def __init__(self, config):
        super().__init__(config)
        self.chunk_size = 16
        self.token_budget = 1024
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # 对k，v 剪枝的模型会用到repeat_kv，标准transformer架构是数目相同的qkv
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # naive torch quest
        # k, v size(): 1,32,1,128
        # quest = True
        # if quest:


        # 计算attn_weights
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        ################################################################################################################################################
        sign = (query_states > 0) + (~(query_states > 0)) * -1
        max_key = key_states * sign
        postive_query = query_states * sign

        # expend max_key to be divisible by chunk_size
        seq_length = max_key.shape[-2]
        padding_length = self.chunk_size - ((seq_length - 1) % self.chunk_size + 1)
        max_key = torch.cat(
            [
                max_key,
                torch.ones(
                    (max_key.shape[0], max_key.shape[1], padding_length, max_key.shape[3]),
                    device=max_key.device,
                )
                * torch.tensor(torch.finfo(max_key.dtype).min),
            ],
            dim=-2,
        )

        # chunk max_key into chunk_size tokens
        chunk_max_key = max_key.reshape(
            max_key.shape[0],
            max_key.shape[1],
            max_key.shape[2] // self.chunk_size,
            self.chunk_size,
            max_key.shape[3],
        ).amax(dim=-2)

        # duplicate chunk_max_key chunk_size times
        chunk_max_key = chunk_max_key.unsqueeze(-2).repeat(1, 1, 1, self.chunk_size, 1)
        # reshape chunk_max_key to the original shape
        chunk_max_key = chunk_max_key.reshape(
            chunk_max_key.shape[0], chunk_max_key.shape[1], -1, chunk_max_key.shape[-1]
        )[:, :, :seq_length, :]

        quantized_weight = torch.matmul(
            postive_query.float(),
            chunk_max_key.transpose(2, 3),
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
            quantized_weight = quantized_weight + attention_mask
            quantized_weight = torch.max(
                quantized_weight, torch.tensor(torch.finfo(quantized_weight.dtype).min)
            )

        token_budget = min(kv_seq_len, self.token_budget)

        attn_weights_for_selection = quantized_weight

        if token_budget > 0:
            mask_bottom = local_heavy_hitter_mask(
                attn_weights_for_selection, token_budget, self.chunk_size
            )  # Default: No padding applied to input
        else:
            mask_bottom = torch.zeros_like(attn_weights_for_selection, dtype=torch.bool)

        mask_bottom = torch.tril(mask_bottom, diagonal=position_ids[0][0].item())
        attn_weights[~mask_bottom] = torch.tensor(torch.finfo(attn_weights.dtype).min)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class CustomConfig(LlamaConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

if __name__ == "__main__":

    device=torch.device('cpu')
    hidden_states = torch.randn(1, 1, 4096, dtype=torch.float, device=device)
    
    config = CustomConfig(
        hidden_size=4096,  # 隐藏层大小
        intermediate_size=11008,  # 中间层大小
        num_hidden_layers=32,  # 隐藏层数
        num_attention_heads=32,  # 注意力头数
        vocab_size=32000,  # 词汇表大小
        max_position_embeddings=2048,  # 最大位置嵌入
        initializer_range=0.02,  # 初始化范围
        use_cache=True,  # 是否使用缓存
        pad_token_id=1,  # 填充标记ID
        bos_token_id=0,  # 开始标记ID
        eos_token_id=2,  # 结束标记ID
        chunk_size=16,
        # 其他配置项...
    )

    attention_layer = CustomLlamaAttention(config).to(device)

    batch_size, seq_length = hidden_states.shape[:2]
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

    attention_layer.forward(hidden_states, position_ids=position_ids)

    print('finished')