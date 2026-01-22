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
def avg_pool2d_backward_shape_check(input, gradOutput, nbatch, kH, kW, dH, dW, padH, padW, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth, mem_format):
    pool2d_shape_check(input, kH, kW, dH, dW, padH, padW, 1, 1, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth, mem_format)
    ndim = input.dim()
    nOutputPlane = nInputPlane
    check_dim_size(gradOutput, ndim, ndim - 3, nOutputPlane)
    check_dim_size(gradOutput, ndim, ndim - 2, outputHeight)
    check_dim_size(gradOutput, ndim, ndim - 1, outputWidth)