import math
from enum import Enum
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._prims_common as utils
from torch import SymBool, SymFloat, Tensor
from torch._decomp import (
from torch._ops import OpOverload
from torch._prims import _prim_elementwise_meta, ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _broadcast_shapes, _maybe_broadcast
from torch.utils import _pytree as pytree
import torch._refs
import torch._refs.nn.functional
import torch._refs.special
@register_meta([aten._scaled_dot_product_flash_attention_backward])
def meta__scaled_dot_product_flash_backward(grad_out: Tensor, query: Tensor, key: Tensor, value: Tensor, out: Tensor, logsumexp: Tensor, cum_seq_q: Tensor, cum_seq_k: Tensor, max_q: int, max_k: int, dropout_p: float, is_causal: bool, philox_seed: Tensor, philox_offset: Tensor, scale: Optional[float]=None):
    if device_hint(query) != 'cpu':
        grad_q = torch.empty_like(query.transpose(1, 2)).transpose(1, 2)
        grad_k = torch.empty_like(key.transpose(1, 2)).transpose(1, 2)
        grad_v = torch.empty_like(value.transpose(1, 2)).transpose(1, 2)
        return (grad_q, grad_k, grad_v)
    batch_size = query.size(0)
    num_heads = query.size(1)
    head_dim = query.size(3)
    len_q = query.size(2) if device_hint(query) == 'cpu' else max_q
    len_k = key.size(2) if device_hint(query) == 'cpu' else max_k
    grad_q = torch.empty_permuted((batch_size, num_heads, len_q, head_dim), (0, 2, 1, 3), dtype=query.dtype, device=query.device)
    grad_k = torch.empty_permuted((batch_size, num_heads, len_k, head_dim), (0, 2, 1, 3), dtype=key.dtype, device=key.device)
    grad_v = torch.empty_permuted((batch_size, num_heads, len_k, head_dim), (0, 2, 1, 3), dtype=value.dtype, device=value.device)
    return (grad_q, grad_k, grad_v)