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
@register_meta([aten.linalg_lu_factor_ex.default, aten.linalg_lu_factor_ex.out])
@out_wrapper('LU', 'pivots', 'info')
def linalg_lu_factor_ex_meta(A: Tensor, *, pivot: bool=True, check_errors: bool=False) -> Tuple[Tensor, Tensor, Tensor]:
    torch._check(A.ndim >= 2, lambda: f'torch.lu_factor: Expected tensor with 2 or more dimensions. Got size: {A.shape} instead')
    sizes = list(A.shape)
    m = sizes[-2]
    n = sizes[-1]
    LU = torch.empty_strided(size=sizes, stride=make_contiguous_strides_for(sizes, row_major=False), dtype=A.dtype, device=A.device)
    sizes.pop()
    sizes[-1] = min(m, n)
    pivots = A.new_empty(sizes, dtype=torch.int)
    sizes.pop()
    info = A.new_empty(sizes, dtype=torch.int)
    return (LU, pivots, info)