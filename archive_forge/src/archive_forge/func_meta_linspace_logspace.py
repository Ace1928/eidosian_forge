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
@register_meta([aten.linspace, aten.logspace])
@out_wrapper()
def meta_linspace_logspace(start, end, steps, base=None, dtype=None, device=None, layout=torch.strided, pin_memory=False, requires_grad=False):
    if isinstance(start, torch.Tensor):
        torch._check(start.dim() == 0, lambda: 'linspace only supports 0-dimensional start and end tensors')
    if isinstance(end, torch.Tensor):
        torch._check(end.dim() == 0, lambda: 'linspace only supports 0-dimensional start and end tensors')
    if any((isinstance(arg, complex) for arg in (start, end, steps))):
        default_complex_dtype = utils.corresponding_complex_dtype(torch.get_default_dtype())
        if dtype is None:
            dtype = default_complex_dtype
        else:
            torch._check(utils.is_complex_dtype(dtype), lambda: f"linspace(): inferred dtype {default_complex_dtype} can't be safely cast to passed dtype {dtype}")
    else:
        dtype = dtype or torch.get_default_dtype()
    assert isinstance(dtype, torch.dtype)
    torch._check_type(isinstance(steps, IntLike), lambda: f'received an invalid combination of arguments - got ({type(start).__name__}, {type(end).__name__}, {type(steps).__name__})')
    assert isinstance(steps, IntLike)
    torch._check(steps >= 0, lambda: 'number of steps must be non-negative')
    return torch.empty((steps,), dtype=dtype, layout=layout, device='meta', pin_memory=pin_memory, requires_grad=requires_grad)