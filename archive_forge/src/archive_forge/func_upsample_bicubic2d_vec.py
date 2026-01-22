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
@register_decomposition(aten.upsample_bicubic2d.vec)
@aten.upsample_bicubic2d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_bicubic2d.vec.py_impl(DispatchKey.Autograd)
@out_wrapper()
@pw_cast_for_opmath
def upsample_bicubic2d_vec(a: Tensor, output_size: Optional[Tuple[int, int]], align_corners: bool, scale_factors: Optional[Tuple[float, float]]=None) -> Tensor:
    torch._check(bool(output_size) + bool(scale_factors) == 1, lambda: 'Must specify exactly one of output_size and scale_factors.')
    if output_size is None:
        assert scale_factors is not None
        output_size = cast(Tuple[int, int], tuple((sym_int(sym_float(w) * scale) for w, scale in zip(a.shape[2:], scale_factors))))
    scale_h, scale_w = scale_factors if scale_factors else (None, None)
    return upsample_bicubic2d_default(a, output_size, align_corners, scale_h, scale_w)