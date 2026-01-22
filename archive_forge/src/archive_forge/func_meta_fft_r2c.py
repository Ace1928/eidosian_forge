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
@register_meta([aten._fft_r2c.default, aten._fft_r2c.out])
@out_wrapper()
def meta_fft_r2c(self, dim, normalization, onesided):
    assert self.dtype.is_floating_point
    output_sizes = list(self.size())
    if onesided:
        last_dim = dim[-1]
        last_dim_halfsize = output_sizes[last_dim] // 2 + 1
        output_sizes[last_dim] = last_dim_halfsize
    return self.new_empty(output_sizes, dtype=utils.corresponding_complex_dtype(self.dtype))