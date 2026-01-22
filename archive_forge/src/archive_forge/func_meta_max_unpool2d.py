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
@register_meta(aten.max_unpool2d)
@out_wrapper()
def meta_max_unpool2d(self_, indices, output_size):
    utils.alert_not_deterministic('max_unpooling2d_forward_out')
    torch._check(indices.dtype == torch.int64, lambda: f'elements in indices should be type int64 but got: {indices.dtype}')
    torch._check(len(output_size) == 2, lambda: f'There should be exactly two elements (height, width) in output_size, but got {len(output_size)} elements.')
    oheight, owidth = output_size
    torch._check(self_.ndim in (3, 4), lambda: f'Input to max_unpooling2d should be a 3d or 4d Tensor, but got a tensor with {self_.ndim} dimensions.')
    torch._check(self_.shape == indices.shape, lambda: f'Expected shape of indices to be same as that of the input tensor ({self_.shape}) but got indices tensor with shape: {indices.shape}')
    for i in range(1, self_.ndim):
        torch._check(self_.size(i) > 0, lambda: f'max_unpooling2d(): Expected input to have non-zero size for non-batch dimensions, but got {self_.shape} with dimension {i} being empty.')
    self = self_.contiguous()
    if self_.ndim == 3:
        nchannels = self.size(0)
        result = self.new_empty((nchannels, oheight, owidth))
    else:
        nbatch = self.size(0)
        nchannels = self.size(1)
        result = self.new_empty((nbatch, nchannels, oheight, owidth))
    return result