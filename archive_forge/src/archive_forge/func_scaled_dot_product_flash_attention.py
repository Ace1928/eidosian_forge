import functools
import numbers
import operator
import sys
from enum import Enum
from functools import partial, reduce
from itertools import chain, product
from typing import Callable, cast, Iterable, List, Optional, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch.nn.functional as F
from torch import sym_float, sym_int, Tensor
from torch._decomp import register_decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import IntLike, NumberType, TensorLike, TensorSequenceType
from torch._prims_common.wrappers import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
@register_decomposition(aten._scaled_dot_product_flash_attention.default)
def scaled_dot_product_flash_attention(query: Tensor, key: Tensor, value: Tensor, dropout_p: float=0.0, is_causal: bool=False, return_debug_mask: bool=False, *, scale: Optional[float]=None) -> Tuple[Tensor, Tensor, Tensor, Tensor, int, int, Tensor, Tensor, Tensor]:
    dtype = query.dtype
    batchSize, num_head, qSize, headSize = (query.shape[0], query.shape[1], query.shape[2], query.shape[3])
    torch._check(torch.is_floating_point(query) and dtype is not torch.half, lambda: f'query must be FP32, FP64, BF16 but got {query.dtype}')
    torch._check(query.dim() == 4 and key.dim() == 4 and (value.dim() == 4), lambda: f'q, k, v must be a 4 dimensional tensor, got {query.dim()}, {key.dim()}, {value.dim()}')
    torch._check(dropout_p == 0.0, lambda: f'dropout probability must be zero, got {dropout_p}')
    torch._check(query.shape[3] == value.shape[3] and key.shape[3] == value.shape[3], lambda: 'q, k, v should have the same head size')
    torch._check(return_debug_mask is False, lambda: 'return_debug_mask is not supported.')
    logsumexp = torch.empty([batchSize, qSize, num_head, headSize], dtype=torch.float)
    cum_seq_q, cum_seq_k = (torch.empty([], dtype=torch.long), torch.empty([], dtype=torch.long))
    max_q, max_k = (0, 0)
    philox_seed, philox_offset = (torch.empty([], dtype=torch.long), torch.empty([], dtype=torch.long))
    debug_attn_mask = torch.empty([], dtype=query.dtype, device=query.device, requires_grad=query.requires_grad)
    output, _ = aten._scaled_dot_product_attention_math.default(query, key, value, None, dropout_p, is_causal, None, scale=scale)
    output = output.transpose(1, 2).contiguous(memory_format=torch.contiguous_format)
    return (output.transpose(1, 2), logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask)