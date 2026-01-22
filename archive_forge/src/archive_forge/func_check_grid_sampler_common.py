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
def check_grid_sampler_common(input: Tensor, grid: Tensor):
    torch._check(input.device == grid.device, lambda: f'grid_sampler(): expected input and grid to be on same device, but input is on {input.device} and grid is on {grid.device}')
    torch._check(input.layout == torch.strided and grid.layout == torch.strided, lambda: f'grid_sampler(): expected input and grid to have torch.strided layout, but input has {input.layout} and grid has {grid.layout}')
    torch._check(input.shape[0] == grid.shape[0], lambda: f'grid_sampler(): expected grid and input to have same batch size, but got input with sizes {input.shape} and grid with sizes {grid.shape}')
    torch._check(grid.shape[-1] == input.ndim - 2, lambda: f'grid_sampler(): expected grid to have size {input.ndim - 2} in last dimension, but got grid with sizes {grid.shape}')
    for i in range(2, input.ndim):
        torch._check(input.shape[i] > 0, lambda: f'grid_sampler(): expected input to have non-empty spatial dimensions, but input has sizes {input.shape} with dimension {i} being empty')