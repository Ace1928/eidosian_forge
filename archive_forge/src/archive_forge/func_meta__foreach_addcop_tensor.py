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
@register_meta([aten._foreach_addcdiv_.Tensor, aten._foreach_addcmul_.Tensor])
def meta__foreach_addcop_tensor(self, tensor1, tensor2, scalars):
    torch._check(all((isinstance(l, List) for l in [self, tensor1, tensor2])) and isinstance(scalars, torch.Tensor), lambda: f'_foreach_addc*_ op expects arguments of type: List[Tensor], List[Tensor], List[Tensor], tensor, but got: {type(self)}, {type(tensor1)}, {type(tensor2)}, and {type(scalars)}')
    torch._check(len(self) > 0, lambda: 'input tensor list must not be empty.')
    torch._check(len(self) == len(tensor1) and len(self) == len(tensor2), lambda: 'All input tensor lists must have the same length')