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
def view_as_complex(self: TensorLikeType) -> TensorLikeType:
    input_dtype = self.dtype
    torch._check(utils.is_float_dtype(input_dtype), lambda: f'view_as_complex is only supported for floating pointtensors, but got a tensor of scalar type: {input_dtype}')
    sizes = self.size()
    torch._check(len(sizes) != 0, lambda: 'Input tensor must have one or more dimensions')
    torch._check(sizes[-1] == 2, lambda: 'Tensor must have a last dimension of size 2')
    old_strides = self.stride()
    torch._check(old_strides[-1] == 1, lambda: 'Tensor must have a last dimension with stride 1')
    dims = old_strides[:-1]
    torch._check(py_all((stride % 2 == 0 for stride in dims)), lambda: 'Tensor must have a stride divisible by 2 for all but last dimension')
    torch._check(self.storage_offset() % 2 == 0, lambda: 'Tensor must have a storage_offset divisible by 2')
    return prims.view_element_type(self, utils.corresponding_complex_dtype(input_dtype)).squeeze(-1)