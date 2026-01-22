import builtins
import collections
import inspect
import itertools
import math
import operator
import warnings
from collections.abc import Iterable
from enum import Enum
from functools import partial, reduce, singledispatch, wraps
from typing import Any, Callable, Dict, List, Optional, overload, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch import sym_float, sym_int
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._decomp import register_decomposition
import torch._refs._conversions
import torch._refs.fft
import torch._refs.linalg
import torch._refs.nn.functional
import torch._refs.special
def sum_to_size(a: Tensor, *shape) -> Tensor:
    shape = utils.extract_shape_from_varargs(shape, validate=False)
    torch._check(utils.is_expandable_to(shape, a.shape), lambda: f'sum_to_size: size "{shape}" is not expandable to size "{a.shape}"')
    if utils.is_same_shape(shape, a.shape) and len(shape) > 0:
        return prims.view_of(a)
    leading_dims = a.ndim - len(shape)
    reduce_dims = tuple(range(leading_dims)) + tuple((i for i in range(leading_dims, len(shape)) if shape[i - leading_dims] == 1 and a.shape[i] != 1))
    return torch.sum(a, dim=reduce_dims, keepdim=True, dtype=None)