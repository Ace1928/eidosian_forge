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
@register_meta(aten.mm)
@out_wrapper()
def meta_mm(a, b):
    torch._check(a.dim() == 2, lambda: 'a must be 2D')
    torch._check(b.dim() == 2, lambda: 'b must be 2D')
    N, M1 = a.shape
    M2, P = b.shape
    torch._check(M1 == M2, lambda: f'a and b must have same reduction dim, but got [{N}, {M1}] X [{M2}, {P}].')
    return a.new_empty(N, P)