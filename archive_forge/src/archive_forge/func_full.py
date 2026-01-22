from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@builtin
def full(shape, value, dtype, _builder=None):
    """
    Returns a tensor filled with the scalar value for the given :code:`shape` and :code:`dtype`.

    :param shape: Shape of the new array, e.g., (8, 16) or (8, )
    :value value: A scalar value to fill the array with
    :type shape: tuple of ints
    :param dtype: Data-type of the new array, e.g., :code:`tl.float16`
    :type dtype: DType
    """
    shape = _shape_check_impl(shape)
    value = _constexpr_to_value(value)
    dtype = _constexpr_to_value(dtype)
    return semantic.full(shape, value, dtype, _builder)