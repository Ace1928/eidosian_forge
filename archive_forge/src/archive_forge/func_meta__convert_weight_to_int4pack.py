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
@register_meta([aten._convert_weight_to_int4pack])
def meta__convert_weight_to_int4pack(w, inner_k_tiles):
    torch._check(w.dim() == 2, lambda: 'w must be a 2D tensor')
    torch._check(w.dtype is torch.int32, lambda: f'expected w to be int32, got {w.dtype}')
    n = w.size(0)
    k = w.size(1)
    return w.new_empty((n // 8, k // (inner_k_tiles * 16), 32, inner_k_tiles // 2), dtype=torch.int32)