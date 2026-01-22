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
@register_meta([aten._flash_attention_forward])
def meta__flash_attention_forward(query: Tensor, key: Tensor, value: Tensor, cum_seq_q: Optional[Tensor], cum_seq_k: Optional[Tensor], max_q: int, max_k: int, dropout_p: float, is_causal: bool, return_debug_mask: bool, scale: Optional[float]=None):
    batch_size = query.size(0)
    max_seqlen_batch_q = query.size(1)
    num_heads = query.size(2)
    head_dim = query.size(3)
    max_seqlen_batch_k = key.size(1)
    attention = torch.empty_like(query)
    logsumexp = torch.empty((batch_size, num_heads, max_seqlen_batch_q), dtype=torch.float, device=query.device)
    if return_debug_mask:
        blocksize_c = 128 if head_dim > 64 else 256
        max_seqlen_k = math.ceil(max_seqlen_batch_q / blocksize_c)
        if max_seqlen_batch_k <= 128:
            max_seqlen_k = 128
        elif max_seqlen_batch_k <= 256:
            max_seqlen_k = 256
        debug_mask = torch.empty((batch_size, num_heads, max_seqlen_batch_q, max_seqlen_k), dtype=query.dtype, device=query.device)
    else:
        debug_mask = torch.empty(0, dtype=query.dtype, device=query.device)
    return (attention, logsumexp, torch.empty((), dtype=torch.long, device='meta'), torch.empty((), dtype=torch.long, device='meta'), debug_mask)