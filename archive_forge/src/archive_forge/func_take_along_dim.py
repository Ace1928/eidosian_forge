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
@out_wrapper()
def take_along_dim(a: torch.Tensor, indices: torch.Tensor, dim: Optional[int]=None) -> torch.Tensor:
    torch._check(a.ndim == indices.ndim, lambda: f'torch.take_along_dim(): input and indices should have the same number of dimensions, but got {a.ndim} dimensions for input, and {indices.ndim} dimensions for indices')
    torch._check(utils.is_integer_dtype(indices.dtype), lambda: f'torch.take_along_dim(): dtype of indices should be int but got {indices.dtype} instead')
    if dim is None:
        return torch.gather(a.view(-1), 0, indices.view(-1))
    else:
        self_sizes = list(a.shape)
        self_sizes[dim] = indices.size(dim)
        broadcast_shape = utils.infer_size_shapes(self_sizes, indices.size())
        indices_broadcast = broadcast_to(indices, broadcast_shape)
        indices_sizes = list(indices.shape)
        indices_sizes[dim] = a.size(dim)
        broadcast_shape = utils.infer_size_shapes(indices_sizes, a.size())
        self_broadcast = broadcast_to(a, broadcast_shape)
        return torch.gather(self_broadcast, dim, indices_broadcast)