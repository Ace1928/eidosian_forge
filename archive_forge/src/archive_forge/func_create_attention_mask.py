import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...file_utils import ModelOutput
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_git import GitConfig, GitVisionConfig
def create_attention_mask(self, tgt, memory, tgt_mask, past_key_values_length, memory_key_padding_mask=None):
    num_tgt = tgt.shape[1]
    num_memory = memory.shape[1]
    device = tgt.device
    dtype = tgt.dtype
    top_left = torch.zeros((num_memory, num_memory), device=device, dtype=dtype)
    top_right = torch.full((num_memory, num_tgt + past_key_values_length), float('-inf'), device=tgt.device, dtype=dtype)
    bottom_left = torch.zeros((num_tgt, num_memory), dtype=dtype, device=tgt_mask.device)
    if past_key_values_length > 0:
        tgt_mask = torch.zeros((tgt_mask.shape[0], tgt_mask.shape[0] + past_key_values_length), dtype=dtype, device=tgt_mask.device)
    left = torch.cat((top_left, bottom_left), dim=0)
    right = torch.cat((top_right, tgt_mask.to(dtype)), dim=0)
    full_attention_mask = torch.cat((left, right), dim=1)[None, :]
    if memory_key_padding_mask is None:
        memory_key_padding_mask = torch.full((memory.shape[0], memory.shape[1]), fill_value=False, device=device)
    if memory_key_padding_mask.dtype != torch.bool:
        raise ValueError('Memory key padding mask must be a boolean tensor.')
    zero_negative_infinity = torch.zeros_like(memory_key_padding_mask, dtype=tgt.dtype)
    zero_negative_infinity[memory_key_padding_mask] = float('-inf')
    full_attention_mask = full_attention_mask.expand((memory_key_padding_mask.shape[0], num_memory + num_tgt, num_memory + past_key_values_length + num_tgt))
    full_attention_mask = full_attention_mask.clone()
    origin_left = full_attention_mask[:, :, :num_memory]
    update = zero_negative_infinity[:, None, :]
    full_attention_mask[:, :, :num_memory] = origin_left + update
    full_attention_mask = full_attention_mask[:, None, :, :]
    return full_attention_mask