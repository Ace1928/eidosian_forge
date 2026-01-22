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
@register_meta(aten.copy_.default)
def meta_copy_(self, src, non_blocking=False):
    if torch._debug_has_internal_overlap(self) == 1:
        raise RuntimeError('more than one element of the written-to tensor refers to a single memory location')
    if isinstance(src, Tensor):
        intermediate = src.to(self, non_blocking)
        if self.size() != intermediate.size():
            aten.expand_copy.default(intermediate, self.size())
    return self