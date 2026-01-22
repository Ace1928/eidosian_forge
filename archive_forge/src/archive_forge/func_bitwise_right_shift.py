from __future__ import annotations
from ._dtypes import (
from ._array_object import Array
import cupy as np
def bitwise_right_shift(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.right_shift <numpy.right_shift>`.

    See its docstring for more information.
    """
    if x1.dtype not in _integer_dtypes or x2.dtype not in _integer_dtypes:
        raise TypeError('Only integer dtypes are allowed in bitwise_right_shift')
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    if np.any(x2._array < 0):
        raise ValueError('bitwise_right_shift(x1, x2) is only defined for x2 >= 0')
    return Array._new(np.right_shift(x1._array, x2._array))