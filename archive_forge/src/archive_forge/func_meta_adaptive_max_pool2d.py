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
@register_meta(aten.adaptive_max_pool2d)
@out_wrapper('out', 'indices')
def meta_adaptive_max_pool2d(input, output_size):
    ndim = input.ndim
    torch._check(ndim in (3, 4), lambda: f'adaptive_max_pool2d(): Expected 3D or 4D tensor, but got: {input.shape}')
    for i in range(1, ndim):
        torch._check(input.size(i) > 0, lambda: f'adaptive_max_pool2d(): Expected input to have non-zero size for non-batch dimensions, but input has sizes {input.shape} with dimension {i} being empty')
    torch._check(len(output_size) == 2, lambda: 'adaptive_max_pool2d(): internal error: output_size.size() must be 2')
    dimH = 1
    sizeB = 1
    sizeD = 0
    if input.ndim == 4:
        sizeB = input.size(0)
        dimH += 1
    sizeD = input.size(dimH - 1)
    osizeH, osizeW = output_size
    if input.ndim == 3:
        out_shape = (sizeD, osizeH, osizeW)
        out = input.new_empty(out_shape)
        indices = input.new_empty(out_shape, dtype=torch.int64)
        return (out, indices)
    else:
        out_shape = (sizeB, sizeD, osizeH, osizeW)
        memory_format = utils.suggest_memory_format(input)
        out = input.new_empty(out_shape).to(memory_format=memory_format)
        indices = input.new_empty(out_shape, dtype=torch.int64).to(memory_format=memory_format)
        return (out, indices)