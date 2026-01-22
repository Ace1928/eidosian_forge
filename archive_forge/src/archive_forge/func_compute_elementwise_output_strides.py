from __future__ import annotations
import operator
import warnings
import weakref
from contextlib import nullcontext
from enum import Enum
from functools import cmp_to_key, reduce
from typing import (
import torch
from torch import sym_float, sym_int, sym_max
def compute_elementwise_output_strides(*tensors) -> Tuple[int, ...]:
    """
    Computes the output strides for elementwise operations.
    """
    if len(tensors) == 0:
        msg = "Can't compute elementwise output strides for zero tensors!"
        raise ValueError(msg)
    check_same_shape(*tensors, allow_cpu_scalar_tensors=True)
    tensors = tuple((a for a in tensors if isinstance(a, TensorLike) and (not is_cpu_scalar_tensor(a))))
    if len(tensors) == 0:
        return ()
    ndim = tensors[0].ndim
    shape = tensors[0].shape
    if ndim == 0:
        return ()
    if ndim == 1:
        return (1,)
    logical_to_physical_perm = compute_elementwise_output_logical_to_physical_perm(*tensors, _skip_checks=True)
    permuted_shape = apply_perm(shape, logical_to_physical_perm)
    new_strides = make_contiguous_strides_for(permuted_shape)
    permuted_strides = apply_perm(new_strides, invert_perm(logical_to_physical_perm))
    return tuple(permuted_strides)