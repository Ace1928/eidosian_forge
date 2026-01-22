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
def check_grid_sampler_3d(input: Tensor, grid: Tensor, interpolation_mode: int):
    torch._check(input.ndim == 5 and input.ndim == grid.ndim, lambda: f'grid_sampler(): expected 5D input and grid with same number of dimensions, but got input with sizes {input.shape} and grid with sizes {grid.shape}')
    torch._check(not (input.ndim == 5 and interpolation_mode == GridSamplerInterpolation.BICUBIC.value), lambda: 'grid_sampler(): bicubic interpolation only supports 4D input')