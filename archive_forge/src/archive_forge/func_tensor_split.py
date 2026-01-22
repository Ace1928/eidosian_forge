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
def tensor_split(a: TensorLikeType, indices_or_sections: Union[Tensor, DimsType], dim: int=0) -> Tuple[TensorLikeType, ...]:
    _dim = utils.canonicalize_dim(a.ndim, dim)
    if a.ndim == 0:
        msg = 'tensor_split: received a rank zero tensor, but expected a tensor of rank one or greater!'
        raise ValueError(msg)
    if isinstance(indices_or_sections, TensorLike):
        if not indices_or_sections.device.type == 'cpu':
            msg = 'tensor_split: if indices_or_sections is a tensor it must be on the CPU, but received one on {}'.format(indices_or_sections.device)
            raise ValueError(msg)
        if indices_or_sections.dtype != torch.long:
            msg = 'tensor_split: if indices_or_sections is a tensor it must have long dtype, '
            f' but received one with dtype {indices_or_sections.dtype}'
            raise ValueError(msg)
    if isinstance(indices_or_sections, IntLike) or (isinstance(indices_or_sections, TensorLike) and indices_or_sections.ndim == 0):
        sections: int = indices_or_sections if isinstance(indices_or_sections, Number) else indices_or_sections.item()
        if sections <= 0:
            msg = f'tensor_split: number of sections must be greater than 0, but was {sections}'
            raise ValueError(msg)
        splits = []
        dim_size = a.shape[_dim]
        min_split_size = math.floor(dim_size / sections)
        num_splits_one_extra = dim_size % sections
        start_idx = 0
        for split_idx in range(sections):
            split_size = min_split_size + 1 if split_idx < num_splits_one_extra else min_split_size
            s = prims.slice_in_dim(a, start_idx, start_idx + split_size, axis=_dim)
            splits.append(s)
            start_idx = start_idx + split_size
        return tuple(splits)
    else:
        indices = indices_or_sections
        if isinstance(indices_or_sections, TensorLike):
            if indices_or_sections.ndim != 1:
                msg = 'tensor_split: non-scalar indices_or_sections tensors must have only one dimension, '
                f'but received a tensor with {indices_or_sections.ndim} dimensions'
                raise ValueError(msg)
            indices = indices_or_sections.tolist()
        splits = []
        start_idx = 0
        for x in indices:
            splits.append(prims.slice_in_dim(a, start_idx, x, axis=_dim))
            start_idx = x
        splits.append(prims.slice_in_dim(a, start_idx, a.shape[_dim], axis=_dim))
        return tuple(splits)