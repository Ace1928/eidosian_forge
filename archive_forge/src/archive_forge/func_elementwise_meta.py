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
def elementwise_meta(*args, type_promotion: ELEMENTWISE_TYPE_PROMOTION_KIND):
    _, result_dtype = utils.elementwise_dtypes(*args, type_promotion_kind=type_promotion)
    args = [_maybe_convert_to_dtype(x, result_dtype) for x in args]
    args = _maybe_broadcast(*args)
    return _prim_elementwise_meta(*args, type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT)