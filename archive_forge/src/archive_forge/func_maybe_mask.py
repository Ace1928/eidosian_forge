import functools
import numbers
import operator
import sys
from enum import Enum
from functools import partial, reduce
from itertools import chain, product
from typing import Callable, cast, Iterable, List, Optional, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch.nn.functional as F
from torch import sym_float, sym_int, Tensor
from torch._decomp import register_decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import IntLike, NumberType, TensorLike, TensorSequenceType
from torch._prims_common.wrappers import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
def maybe_mask(vals, length, range_max, adaptive, dim):
    if isinstance(length, IntLike):
        return (vals, length)
    else:
        assert dim < 0
        mask = range_max >= length.unsqueeze(-1)
        if dim == -2:
            mask = _unsqueeze_to_dim(mask, 4)
        vals = torch.masked_fill(vals, mask, 0.0)
        length = _unsqueeze_to_dim(length, -dim)
        return (vals, length)