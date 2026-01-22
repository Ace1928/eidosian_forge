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
@register_meta([aten._int_mm])
@out_wrapper()
def meta__int_mm(a, b):
    torch._check(a.dim() == 2, lambda: 'a must be a 2D tensor')
    torch._check(b.dim() == 2, lambda: 'b must be a 2D tensor')
    torch._check(a.dtype is torch.int8, lambda: f'expected self to be int8, got {a.dtype}')
    torch._check(b.dtype is torch.int8, lambda: f'expected mat2 to be int8, got {b.dtype}')
    torch._check(a.size(1) == b.size(0), lambda: f'Incompatible matrix sizes for _int_mm ({a.size(0)}x{a.size(1)} and {b.size(0)}x{b.size(1)})')
    return a.new_empty((a.size(0), b.size(1)), dtype=torch.int32)