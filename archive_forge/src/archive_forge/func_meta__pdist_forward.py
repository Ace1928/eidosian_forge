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
@register_meta(aten._pdist_forward)
@out_wrapper()
def meta__pdist_forward(self: Tensor, p: float=2) -> Tensor:
    torch._check(self.is_contiguous(), lambda: '_pdist_forward requires contiguous input')
    n = self.size(0)
    if n <= 1:
        return self.new_empty([0]).to(memory_format=torch.legacy_contiguous_format)
    else:
        return self.new_empty((n * (n - 1) // 2,)).to(memory_format=torch.legacy_contiguous_format)