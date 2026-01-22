import math
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.cpp_extension import load
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_mra import MraConfig
def mra2_attention(query, key, value, mask, num_blocks, approx_mode, block_size=32, initial_prior_first_n_blocks=0, initial_prior_diagonal_n_blocks=0):
    """
    Use Mra to approximate self-attention.
    """
    if mra_cuda_kernel is None:
        return torch.zeros_like(query).requires_grad_()
    batch_size, num_head, seq_len, head_dim = query.size()
    meta_batch = batch_size * num_head
    if seq_len % block_size != 0:
        raise ValueError('sequence length must be divisible by the block_size.')
    num_block_per_row = seq_len // block_size
    query = query.reshape(meta_batch, seq_len, head_dim)
    key = key.reshape(meta_batch, seq_len, head_dim)
    value = value.reshape(meta_batch, seq_len, head_dim)
    if mask is not None:
        query = query * mask[:, :, None]
        key = key * mask[:, :, None]
        value = value * mask[:, :, None]
    if approx_mode == 'full':
        low_resolution_logit, token_count, low_resolution_logit_row_max, value_hat = get_low_resolution_logit(query, key, block_size, mask, value)
    elif approx_mode == 'sparse':
        with torch.no_grad():
            low_resolution_logit, token_count, low_resolution_logit_row_max, _ = get_low_resolution_logit(query, key, block_size, mask)
    else:
        raise Exception('approx_mode must be "full" or "sparse"')
    with torch.no_grad():
        low_resolution_logit_normalized = low_resolution_logit - low_resolution_logit_row_max
        indices, high_resolution_mask = get_block_idxes(low_resolution_logit_normalized, num_blocks, approx_mode, initial_prior_first_n_blocks, initial_prior_diagonal_n_blocks)
    high_resolution_logit = MraSampledDenseMatMul.operator_call(query, key, indices, block_size=block_size) / math.sqrt(head_dim)
    max_vals, max_vals_scatter = sparse_max(high_resolution_logit, indices, num_block_per_row, num_block_per_row)
    high_resolution_logit = high_resolution_logit - max_vals_scatter
    if mask is not None:
        high_resolution_logit = high_resolution_logit - 10000.0 * (1 - sparse_mask(mask, indices)[:, :, :, None])
    high_resolution_attn = torch.exp(high_resolution_logit)
    high_resolution_attn_out = MraSparseDenseMatMul.operator_call(high_resolution_attn, indices, value, num_block_per_row)
    high_resolution_normalizer = MraReduceSum.operator_call(high_resolution_attn, indices, num_block_per_row, num_block_per_row)
    if approx_mode == 'full':
        low_resolution_attn = torch.exp(low_resolution_logit - low_resolution_logit_row_max - 10000.0 * high_resolution_mask) * token_count[:, None, :]
        low_resolution_attn_out = torch.matmul(low_resolution_attn, value_hat)[:, :, None, :].repeat(1, 1, block_size, 1).reshape(meta_batch, seq_len, head_dim)
        low_resolution_normalizer = low_resolution_attn.sum(dim=-1)[:, :, None].repeat(1, 1, block_size).reshape(meta_batch, seq_len)
        log_correction = low_resolution_logit_row_max.repeat(1, 1, block_size).reshape(meta_batch, seq_len) - max_vals
        if mask is not None:
            log_correction = log_correction * mask
        low_resolution_corr = torch.exp(log_correction * (log_correction <= 0).float())
        low_resolution_attn_out = low_resolution_attn_out * low_resolution_corr[:, :, None]
        low_resolution_normalizer = low_resolution_normalizer * low_resolution_corr
        high_resolution_corr = torch.exp(-log_correction * (log_correction > 0).float())
        high_resolution_attn_out = high_resolution_attn_out * high_resolution_corr[:, :, None]
        high_resolution_normalizer = high_resolution_normalizer * high_resolution_corr
        context_layer = (high_resolution_attn_out + low_resolution_attn_out) / (high_resolution_normalizer[:, :, None] + low_resolution_normalizer[:, :, None] + 1e-06)
    elif approx_mode == 'sparse':
        context_layer = high_resolution_attn_out / (high_resolution_normalizer[:, :, None] + 1e-06)
    else:
        raise Exception('config.approx_mode must be "full" or "sparse"')
    if mask is not None:
        context_layer = context_layer * mask[:, :, None]
    context_layer = context_layer.reshape(batch_size, num_head, seq_len, head_dim)
    return context_layer