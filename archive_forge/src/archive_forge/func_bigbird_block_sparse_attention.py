import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_big_bird import BigBirdConfig
def bigbird_block_sparse_attention(self, query_layer, key_layer, value_layer, band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask, n_heads, n_rand_blocks, attention_head_size, from_block_size, to_block_size, batch_size, from_seq_len, to_seq_len, seed, plan_from_length, plan_num_rand_blocks, output_attentions):
    if from_seq_len // from_block_size != to_seq_len // to_block_size:
        raise ValueError('Error the number of blocks needs to be same!')
    rsqrt_d = 1 / math.sqrt(attention_head_size)
    bsz = batch_size
    attn_mask_penalty = -10000.0
    np.random.seed(seed)
    if from_seq_len in [1024, 3072, 4096]:
        rand_attn = [self._bigbird_block_rand_mask(self.max_seqlen, self.max_seqlen, from_block_size, to_block_size, n_rand_blocks, last_idx=1024)[:from_seq_len // from_block_size - 2] for _ in range(n_heads)]
    else:
        if plan_from_length is None:
            plan_from_length, plan_num_rand_blocks = self._get_rand_attn_plan(from_seq_len, from_block_size, n_rand_blocks)
        rand_attn = self._bigbird_block_rand_mask_with_head(from_seq_length=from_seq_len, to_seq_length=to_seq_len, from_block_size=from_block_size, to_block_size=to_block_size, num_heads=n_heads, plan_from_length=plan_from_length, plan_num_rand_blocks=plan_num_rand_blocks)
    rand_attn = np.stack(rand_attn, axis=0)
    rand_attn = torch.tensor(rand_attn, device=query_layer.device, dtype=torch.long)
    rand_attn.unsqueeze_(0)
    rand_attn = torch.cat([rand_attn for _ in range(batch_size)], dim=0)
    rand_mask = self._create_rand_mask_from_inputs(from_blocked_mask, to_blocked_mask, rand_attn, n_heads, n_rand_blocks, bsz, from_seq_len, from_block_size)
    blocked_query_matrix = query_layer.view(bsz, n_heads, from_seq_len // from_block_size, from_block_size, -1)
    blocked_key_matrix = key_layer.view(bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1)
    blocked_value_matrix = value_layer.view(bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1)
    gathered_key = self.torch_gather_b2(blocked_key_matrix, rand_attn)
    gathered_key = gathered_key.view(bsz, n_heads, to_seq_len // to_block_size - 2, n_rand_blocks * to_block_size, -1)
    gathered_value = self.torch_gather_b2(blocked_value_matrix, rand_attn)
    gathered_value = gathered_value.view(bsz, n_heads, to_seq_len // to_block_size - 2, n_rand_blocks * to_block_size, -1)
    first_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, 0], key_layer, ndim=4)
    first_product = first_product * rsqrt_d
    first_product += (1.0 - to_mask) * attn_mask_penalty
    first_attn_weights = nn.functional.softmax(first_product, dim=-1)
    first_context_layer = self.torch_bmm_nd(first_attn_weights, value_layer, ndim=4)
    first_context_layer.unsqueeze_(2)
    second_key_mat = torch.cat([blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, 1], blocked_key_matrix[:, :, 2], blocked_key_matrix[:, :, -1], gathered_key[:, :, 0]], dim=2)
    second_value_mat = torch.cat([blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, 1], blocked_value_matrix[:, :, 2], blocked_value_matrix[:, :, -1], gathered_value[:, :, 0]], dim=2)
    second_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, 1], second_key_mat, ndim=4)
    second_seq_pad = torch.cat([to_mask[:, :, :, :3 * to_block_size], to_mask[:, :, :, -to_block_size:], to_mask.new_ones([bsz, 1, 1, n_rand_blocks * to_block_size])], dim=3)
    second_rand_pad = torch.cat([rand_mask.new_ones([bsz, n_heads, from_block_size, 4 * to_block_size]), rand_mask[:, :, 0]], dim=3)
    second_product = second_product * rsqrt_d
    second_product += (1.0 - torch.minimum(second_seq_pad, second_rand_pad)) * attn_mask_penalty
    second_attn_weights = nn.functional.softmax(second_product, dim=-1)
    second_context_layer = self.torch_bmm_nd(second_attn_weights, second_value_mat, ndim=4)
    second_context_layer.unsqueeze_(2)
    exp_blocked_key_matrix = torch.cat([blocked_key_matrix[:, :, 1:-3], blocked_key_matrix[:, :, 2:-2], blocked_key_matrix[:, :, 3:-1]], dim=3)
    exp_blocked_value_matrix = torch.cat([blocked_value_matrix[:, :, 1:-3], blocked_value_matrix[:, :, 2:-2], blocked_value_matrix[:, :, 3:-1]], dim=3)
    middle_query_matrix = blocked_query_matrix[:, :, 2:-2]
    inner_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, exp_blocked_key_matrix, ndim=5)
    inner_band_product = inner_band_product * rsqrt_d
    rand_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, gathered_key[:, :, 1:-1], ndim=5)
    rand_band_product = rand_band_product * rsqrt_d
    first_band_product = torch.einsum('bhlqd,bhkd->bhlqk', middle_query_matrix, blocked_key_matrix[:, :, 0])
    first_band_product = first_band_product * rsqrt_d
    last_band_product = torch.einsum('bhlqd,bhkd->bhlqk', middle_query_matrix, blocked_key_matrix[:, :, -1])
    last_band_product = last_band_product * rsqrt_d
    inner_band_product += (1.0 - band_mask) * attn_mask_penalty
    first_band_product += (1.0 - to_mask[:, :, :, :to_block_size].unsqueeze(3)) * attn_mask_penalty
    last_band_product += (1.0 - to_mask[:, :, :, -to_block_size:].unsqueeze(3)) * attn_mask_penalty
    rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * attn_mask_penalty
    band_product = torch.cat([first_band_product, inner_band_product, rand_band_product, last_band_product], dim=-1)
    attn_weights = nn.functional.softmax(band_product, dim=-1)
    context_layer = self.torch_bmm_nd(attn_weights[:, :, :, :, to_block_size:4 * to_block_size], exp_blocked_value_matrix, ndim=5)
    context_layer += self.torch_bmm_nd(attn_weights[:, :, :, :, 4 * to_block_size:-to_block_size], gathered_value[:, :, 1:-1], ndim=5)
    context_layer += torch.einsum('bhlqk,bhkd->bhlqd', attn_weights[:, :, :, :, :to_block_size], blocked_value_matrix[:, :, 0])
    context_layer += torch.einsum('bhlqk,bhkd->bhlqd', attn_weights[:, :, :, :, -to_block_size:], blocked_value_matrix[:, :, -1])
    second_last_key_mat = torch.cat([blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, -3], blocked_key_matrix[:, :, -2], blocked_key_matrix[:, :, -1], gathered_key[:, :, -1]], dim=2)
    second_last_value_mat = torch.cat([blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, -3], blocked_value_matrix[:, :, -2], blocked_value_matrix[:, :, -1], gathered_value[:, :, -1]], dim=2)
    second_last_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, -2], second_last_key_mat, ndim=4)
    second_last_seq_pad = torch.cat([to_mask[:, :, :, :to_block_size], to_mask[:, :, :, -3 * to_block_size:], to_mask.new_ones([bsz, 1, 1, n_rand_blocks * to_block_size])], dim=3)
    second_last_rand_pad = torch.cat([rand_mask.new_ones([bsz, n_heads, from_block_size, 4 * to_block_size]), rand_mask[:, :, -1]], dim=3)
    second_last_product = second_last_product * rsqrt_d
    second_last_product += (1.0 - torch.minimum(second_last_seq_pad, second_last_rand_pad)) * attn_mask_penalty
    second_last_attn_weights = nn.functional.softmax(second_last_product, dim=-1)
    second_last_context_layer = self.torch_bmm_nd(second_last_attn_weights, second_last_value_mat, ndim=4)
    second_last_context_layer.unsqueeze_(2)
    last_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, -1], key_layer, ndim=4)
    last_product = last_product * rsqrt_d
    last_product += (1.0 - to_mask) * attn_mask_penalty
    last_attn_weights = nn.functional.softmax(last_product, dim=-1)
    last_context_layer = self.torch_bmm_nd(last_attn_weights, value_layer, ndim=4)
    last_context_layer.unsqueeze_(2)
    context_layer = torch.cat([first_context_layer, second_context_layer, context_layer, second_last_context_layer, last_context_layer], dim=2)
    context_layer = context_layer.view((bsz, n_heads, from_seq_len, -1)) * from_mask
    context_layer = torch.transpose(context_layer, 1, 2)
    if output_attentions:
        attention_probs = torch.zeros(bsz, n_heads, from_seq_len, to_seq_len, dtype=torch.float, device=context_layer.device)
        attention_probs[:, :, :from_block_size, :] = first_attn_weights
        attention_probs[:, :, from_block_size:2 * from_block_size, :3 * to_block_size] = second_attn_weights[:, :, :, :3 * to_block_size]
        attention_probs[:, :, from_block_size:2 * from_block_size, -to_block_size:] = second_attn_weights[:, :, :, 3 * to_block_size:4 * to_block_size]
        for p1, i1, w1 in zip(range(bsz), rand_attn, second_attn_weights):
            for p2, i2, w2 in zip(range(n_heads), i1, w1):
                attn_probs_view = attention_probs.view(bsz, n_heads, from_seq_len // from_block_size, from_block_size, to_seq_len // to_block_size, to_block_size)
                right_slice = w2[:, 4 * to_block_size:]
                attn_probs_view[p1, p2, 1, :, i2[0]] = right_slice.view(from_block_size, n_rand_blocks, to_block_size)
        for q_idx in range(from_seq_len // from_block_size - 4):
            attn_probs_view = attention_probs.view(bsz, n_heads, from_seq_len // from_block_size, from_block_size, to_seq_len // to_block_size, to_block_size)[:, :, 2:-2, :, 1:-1, :]
            right_slice = attn_weights[:, :, q_idx, :, to_block_size:4 * to_block_size]
            attn_probs_view[:, :, q_idx, :, q_idx:q_idx + 3, :] = right_slice.view(bsz, n_heads, from_block_size, 3, to_block_size)
        attention_probs[:, :, 2 * from_block_size:-2 * from_block_size, :to_block_size] = attn_weights[:, :, :, :, :to_block_size].view(bsz, n_heads, -1, to_block_size)
        attention_probs[:, :, 2 * from_block_size:-2 * from_block_size, -to_block_size:] = attn_weights[:, :, :, :, -to_block_size:].view(bsz, n_heads, -1, to_block_size)
        for p1, i1, w1 in zip(range(bsz), rand_attn, attn_weights):
            for p2, i2, w2 in zip(range(n_heads), i1, w1):
                for q_idx in range(1, len(i2) - 1):
                    attn_probs_view = attention_probs.view(bsz, n_heads, from_seq_len // from_block_size, from_block_size, to_seq_len // to_block_size, to_block_size)
                    right_slice = w2[q_idx - 1, :, 4 * to_block_size:-to_block_size]
                    attn_probs_view[p1, p2, q_idx + 1, :, i2[q_idx]] = right_slice.view(from_block_size, n_rand_blocks, to_block_size)
        attention_probs[:, :, -2 * from_block_size:-from_block_size, :to_block_size] = second_last_attn_weights[:, :, :, :to_block_size]
        attention_probs[:, :, -2 * from_block_size:-from_block_size, -3 * to_block_size:] = second_last_attn_weights[:, :, :, to_block_size:4 * to_block_size]
        for p1, i1, w1 in zip(range(bsz), rand_attn, second_last_attn_weights):
            for p2, i2, w2 in zip(range(n_heads), i1, w1):
                attn_probs_view = attention_probs.view(bsz, n_heads, from_seq_len // from_block_size, from_block_size, to_seq_len // to_block_size, to_block_size)
                right_slice = w2[:, 4 * to_block_size:]
                attn_probs_view[p1, p2, -2, :, i2[-1]] = right_slice.view(from_block_size, n_rand_blocks, to_block_size)
        attention_probs[:, :, -from_block_size:, :] = last_attn_weights
    else:
        attention_probs = None
    return (context_layer, attention_probs)