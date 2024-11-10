import math
import numpy as np
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast

import types

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)


def sparse_attention_fixed_mask(q, k, mask, v):
    """
    实现稀疏注意力机制，当不同头部中 mask 为 True 的数量相同。

    参数：
    - q: 查询张量，形状 (B, H, Q, D)
    - k: 键张量，形状 (B, H, K, D)
    - mask: 注意力掩码，形状 (B, H, Q, K)
    - v: 值张量，形状 (B, H, K, D_v)

    返回：
    - output: 注意力输出，形状 (B, H, Q, D_v)
    """

    B, H, Q, D = q.size()
    _, _, K, D_v = v.size()

    # 展平批次和头部维度
    q = q.view(B * H, Q, D)         # (B*H, Q, D)
    k = k.view(B * H, K, D)         # (B*H, K, D)
    v = v.view(B * H, K, D_v)       # (B*H, K, D_v)
    mask = mask.view(B * H, Q, K)   # (B*H, Q, K)

    # 假设 Q=1
    mask = mask.squeeze(1)           # (B*H, K)  

    # 确保每个头部中 mask 为 True 的数量相同
    mask_counts = mask.float().sum(dim=1)
    assert torch.all(mask_counts == mask_counts[0]), "不同头部中 mask 为 True 的数量不相同"
    M = int(mask_counts[0].item())  # 每个头部中 mask 为 True 的数量

    # 获取每个头部中 mask 为 True 的键的索引
    # 使用 topk 获取任意 M 个 True 的位置
    # mask 为 True 的位置用 1.0，False 用 0.0
    # topk 会选择前 M 个 1.0 对应的索引
    indices = mask.float().topk(M, dim=1, largest=True, sorted=False).indices  # (B*H, M)

    # 生成批次索引
    batch_indices = torch.arange(B * H).unsqueeze(1).to(q.device).expand(-1, M)  # (B*H, M)

    # 选择对应的键向量和值向量
    k_masked = k[batch_indices, indices]  # (B*H, M, D)
    v_masked = v[batch_indices, indices]  # (B*H, M, D_v)

    # 确定 k 的最小值（填充值）
    # if k.dtype.is_floating_point:
    #     min_value = torch.finfo(k.dtype).min
    # else:
    #     min_value = torch.iinfo(k.dtype).min

    # 生成 padding_mask，检查每个选中的 k 是否为填充值
    # 假设填充值的所有 D 维度都被设置为 min_value
    # padding_mask = (k_masked == min_value).all(dim=2)  # (B*H, M)

    # 计算点积得分
    # q: (B*H, Q=1, D)
    # k_masked: (B*H, M, D)
    # scores: (B*H, Q, M)
    # print(q.dtype)
    # print(k_masked.dtype)
    scores = torch.bmm(q.float(), k_masked.transpose(1, 2).float())  # (B*H, Q, M)

    scaling_factor = torch.sqrt(torch.tensor(D, dtype=torch.float32, device=q.device))
    scores = scores / scaling_factor

    # 屏蔽填充位置的得分，将其设置为 -inf
    # padding_mask: (B*H, M) -> (B*H, 1, M)
    # scores = scores.masked_fill(padding_mask.unsqueeze(1), float('-inf'))

    # 计算注意力权重
    attn_weights = torch.softmax(scores, dim=-1)  # (B*H, Q, M)

    # 计算注意力输出
    # attn_weights: (B*H, Q, M)
    # v_masked: (B*H, M, D_v)
    # output: (B*H, Q, D_v)
    # print(attn_weights.dtype) 
    # print(v_masked.dtype) 
    output = torch.bmm(attn_weights, v_masked.float())   # (B*H, Q, D_v)

    # 恢复批次和头部维度
    output = output.view(B, H, Q, D_v)            # (B, H, Q, D_v)

    return output


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
    ).amax(dim=-1) # 重新做了一遍之前的操作

    _, topk = chunk_attn_weights.topk(
        k=min(max(3, token_budget // chunk_size), chunk_attn_weights.size(-1)), dim=-1
    ) # 1,32,1,128 选了index
    # repeat topk chunk_size times and recover the original indexes (* chunk_size + arange(chunk_size))
    topk = topk.unsqueeze(-1).repeat(
        1, 1, 1, 1, chunk_size
    ) * chunk_size + torch.arange(chunk_size, device=topk.device) # 恢复原始索引，乘chunk_size再加arange
    topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)
    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool) # (1,32,1,2048)
    mask_bottom.scatter_(-1, topk, True) # 索引位置设置为true

    # remove the padding
    mask_bottom = mask_bottom[:, :, :, :seq_length] #去掉padding

    return mask_bottom


def quest_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size() # 1, 1

    if q_len > 1 or self.layer_id < 2:
        return self.flash_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            **kwargs,
        )

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    ) # (1,32,1,128)
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    ) # (1,32,1,128)
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, position_ids.to(value_states.device))
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups) # (1,32,4606,128)
    value_states = repeat_kv(value_states, self.num_key_value_groups) 
    
    seq_length = key_states.shape[-2]
    padding_length = self.chunk_size - ((seq_length - 1) % self.chunk_size + 1)
    q_sign = query_states > 0
    q_sign_expanded = q_sign.expand(-1, -1, padding_length, -1)
    if query_states.dtype == torch.bfloat16:
        min_value = -1e20
        max_value = 1e20
    elif query_states.dtype == torch.float16:
        min_value = -1e4
        max_value = 1e4
    padding = torch.where(
        q_sign_expanded,
        torch.full_like(key_states[:, :, :1, :].expand(-1, -1, padding_length, -1), min_value),
        torch.full_like(key_states[:, :, :1, :].expand(-1, -1, padding_length, -1), max_value)
    ).to(key_states.device)
    pad_key = torch.cat([key_states, padding], dim=2)
    pad_value = torch.cat(
        [
            value_states,
            torch.zeros(
                (value_states.shape[0], value_states.shape[1], padding_length, value_states.shape[3]),
                device=value_states.device,
                dtype=value_states.dtype
            )
        ],
        dim=-2,
    )
    chunk_pad_key = pad_key.reshape(
        pad_key.shape[0],
        pad_key.shape[1],
        pad_key.shape[2] // self.chunk_size,
        self.chunk_size,
        pad_key.shape[3],
    )
    
    device = chunk_pad_key.device
    total_size = pad_key.shape[-2]

    if self.token_budget >= seq_length:
        quest_need_estimate = False
    else:
        quest_need_estimate = True

    if self.attn_type == 'quest' and quest_need_estimate == True:
        key_min, _ = chunk_pad_key.min(dim=3)  # 形状为 (1, 32, num_chunks, 128)
        key_max, _ = chunk_pad_key.max(dim=3)
        mul_min = key_min * query_states  
        mul_max = key_max * query_states 
        max_mul = torch.max(mul_min, mul_max)  
        chunk_score = max_mul.sum(dim=-1)
        _, topk_indices = torch.topk(chunk_score,self.token_budget // self.chunk_size,dim=-1)

        range_vec = torch.arange(self.chunk_size, device=device).view(1, 1, 1, self.chunk_size)
        topk_indices_expanded = topk_indices.unsqueeze(-1)
        expanded_indices = topk_indices_expanded * self.chunk_size + range_vec
        expanded_indices = expanded_indices.view(bsz, self.num_heads, -1)
        mask = torch.zeros(bsz, self.num_heads, 1, total_size, device=device, dtype=torch.float32)
        scatter_indices = expanded_indices.unsqueeze(2)
        src = torch.ones_like(scatter_indices, dtype=mask.dtype)
        mask.scatter_(dim=-1, index=scatter_indices, src=src)
        # mask = mask[:,:,:,:seq_length]
    elif self.attn_type == 'base':
        mask = torch.ones((bsz, self.num_heads, 1, total_size), device=device, dtype=torch.float32)
    elif self.attn_type == 'sign':
        q_sign = q_sign.int()
        k_sign = (pad_key > 0).int()
        if self.q_ratio < 1.0:
            abs_q = query_states.abs()
            q_topk_indices = abs_q.topk(k=int(query_states.shape[-1]*self.q_ratio),dim=-1).indices
            q_sign_selected =  q_sign.gather(dim=-1, index=q_topk_indices)
            k_topk_indices = q_topk_indices.expand(-1, -1, k_sign.shape[2], -1)
            k_sign_selected = k_sign.gather(dim=-1, index=k_topk_indices)
        else:
            q_sign_selected = q_sign
            k_sign_selected = k_sign
        
        if self.xor == True:
            xor_result = ~(q_sign_selected ^ k_sign_selected)  # (1,32,4008,128)
            scores = xor_result.sum(dim=-1).unsqueeze(2)
        else:
            scores = torch.matmul(q_sign_selected.float(), k_sign_selected.float().transpose(-2,-1))
        if self.topk_ratio is not None:
            topk_values, topk_indices = torch.topk(scores, int(pad_key.shape[2]*self.topk_ratio), dim=-1)
        else:
            raise ValueError("topk_ratio cannot be None when use type 'sign'.")
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, topk_indices, 1)

    attn_output = sparse_attention_fixed_mask(query_states,pad_key,mask,pad_value).to(query_states.dtype)

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

global layer_id
layer_id = 32

def enable_quest_attention_eval(model, args):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_quest_attention_eval(
                module,
                args,
            )

        global layer_id
        if isinstance(module, LlamaAttention):
            # For longchat model
            layer_id -= 1
            model._modules[name].layer_id = layer_id
            model._modules[name].flash_forward = model._modules[name].forward
            model._modules[name].forward = types.MethodType(
                quest_forward, model._modules[name]
            )

            model._modules[name].token_budget = args.token_budget
            model._modules[name].chunk_size = args.chunk_size
            model._modules[name].attn_type = args.attn_type
            model._modules[name].topk_ratio = args.topk_ratio
            model._modules[name].xor = args.xor
            model._modules[name].q_ratio = args.q_ratio