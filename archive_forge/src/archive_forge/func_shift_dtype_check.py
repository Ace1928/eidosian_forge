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
def shift_dtype_check(fn_name, self, val):
    torch._check(utils.is_integer_dtype(self.dtype), lambda: f'{fn_name}: Expected input tensor to have an integral dtype. Got {self.dtype}')
    if isinstance(val, torch.Tensor):
        torch._check(utils.is_integer_dtype(val.dtype), lambda: f'{fn_name}: Expected shift value to have an integral dtype. Got {val.dtype}')
    else:
        torch._check(isinstance(val, IntLike), lambda: f'{fn_name}: Expected shift value to be an int. Got {val}')