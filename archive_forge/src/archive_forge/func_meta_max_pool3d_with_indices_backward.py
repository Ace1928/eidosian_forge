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
@register_meta(aten.max_pool3d_with_indices_backward)
@out_wrapper('grad_input')
def meta_max_pool3d_with_indices_backward(grad_output, input, kernel_size, stride, padding, dilation, ceil_mode, indices):
    torch._check(len(kernel_size) in (1, 3), lambda: 'max_pool3d: kernel_size must either be a single int, or a tuple of three ints')
    kT = kernel_size[0]
    kH = kT if len(kernel_size) == 1 else kernel_size[1]
    kW = kT if len(kernel_size) == 1 else kernel_size[2]
    torch._check(not stride or len(stride) in (1, 3), lambda: 'max_pool3d: stride must either be omitted, a single int, or a tuple of three ints')
    dT = kT if not stride else stride[0]
    dH = kH if not stride else dT if len(stride) == 1 else stride[1]
    dW = kW if not stride else dT if len(stride) == 1 else stride[2]
    torch._check(len(padding) in (1, 3), lambda: 'max_pool3d: padding must either be a single int, or a tuple of three ints')
    pT = padding[0]
    pH = pT if len(padding) == 1 else padding[1]
    pW = pT if len(padding) == 1 else padding[2]
    torch._check(len(dilation) in (1, 3), lambda: 'max_pool3d: dilation must be either a single int, or a tuple of three ints')
    dilationT = dilation[0]
    dilationH = dilationT if len(dilation) == 1 else dilation[1]
    dilationW = dilationT if len(dilation) == 1 else dilation[2]
    torch._check(input.ndim in (4, 5), lambda: 'non-empty 4D or 5D (batch mode) tensor expected for input')
    nslices = input.size(-4)
    itime = input.size(-3)
    iheight = input.size(-2)
    iwidth = input.size(-1)
    otime = grad_output.size(-3)
    oheight = grad_output.size(-2)
    owidth = grad_output.size(-1)
    max_pool3d_backward_shape_check(input, grad_output, indices, nslices, kT, kH, kW, dT, dH, dW, pT, pH, pW, dilationT, dilationH, dilationW, itime, iheight, iwidth, otime, oheight, owidth, 'max_pool3d_with_indices_backward()')
    channels_last = input.ndim == 5 and utils.suggest_memory_format(input) == torch.channels_last_3d
    if input.ndim == 4:
        input_channels_last_check = input.unsqueeze(0)
        channels_last = not input_channels_last_check.is_contiguous() and input_channels_last_check.is_contiguous(memory_format=torch.channels_last_3d)
    grad_input = input.new_empty(input.shape)
    if channels_last:
        grad_input = grad_input.to(memory_format=torch.channels_last_3d)
    return grad_input