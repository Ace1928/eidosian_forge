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
@register_meta(aten.avg_pool3d)
@out_wrapper()
def meta_avg_pool3d(input, kernel_size, stride=(), padding=(0,), ceil_mode=False, count_include_pad=True, divisor_override=None):
    torch._check(len(kernel_size) in (1, 3), lambda: 'avg_pool3d: kernel_size must be a single int, or a tuple of three ints')
    kT = kernel_size[0]
    kH = kT if len(kernel_size) == 1 else kernel_size[1]
    kW = kT if len(kernel_size) == 1 else kernel_size[2]
    torch._check(not stride or len(stride) in (1, 3), lambda: 'avg_pool3d: stride must be omitted, a single int, or a tuple of three ints')
    dT = kT if not stride else stride[0]
    dH = kH if not stride else dT if len(stride) == 1 else stride[1]
    dW = kW if not stride else dT if len(stride) == 1 else stride[2]
    torch._check(len(padding) in (1, 3), lambda: 'avg_pool3d: padding must be a single int, or a tuple of three ints')
    padT = padding[0]
    padH = padT if len(padding) == 1 else padding[1]
    padW = padT if len(padding) == 1 else padding[2]
    torch._check(input.ndim in (4, 5), lambda: 'non-empty 4D or 5D (batch mode) tensor expected for input')
    torch._check(not divisor_override or divisor_override != 0, lambda: 'divisor must be not zero')
    nbatch = input.size(0)
    nslices = input.size(-4)
    itime = input.size(-3)
    iheight = input.size(-2)
    iwidth = input.size(-1)
    otime = pooling_output_shape(itime, kT, padT, dT, 1, ceil_mode)
    oheight = pooling_output_shape(iheight, kH, padH, dH, 1, ceil_mode)
    owidth = pooling_output_shape(iwidth, kW, padW, dW, 1, ceil_mode)
    pool3d_shape_check(input, nslices, kT, kH, kW, dT, dH, dW, padT, padH, padW, 1, 1, 1, itime, iheight, iwidth, otime, oheight, owidth, 'avg_pool3d()', check_input_size=True)
    if input.ndim == 4:
        return input.new_empty((nslices, otime, oheight, owidth))
    else:
        return input.new_empty((nbatch, nslices, otime, oheight, owidth))