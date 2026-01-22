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
def register_meta_foreach(ops):

    def wrapper(fn):

        def register(op):
            op_name = str(op).split('.')[1]
            scalar_op = getattr(aten, op_name.replace('_foreach_', ''))
            _add_op_to_registry(meta_table, op, partial(fn, _scalar_op=scalar_op))
        pytree.tree_map_(register, ops)
        return fn
    return wrapper