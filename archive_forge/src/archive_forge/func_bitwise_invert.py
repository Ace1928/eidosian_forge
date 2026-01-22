from __future__ import annotations
from ._dtypes import (
from ._array_object import Array
import cupy as np
def bitwise_invert(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.invert <numpy.invert>`.

    See its docstring for more information.
    """
    if x.dtype not in _integer_or_boolean_dtypes:
        raise TypeError('Only integer or boolean dtypes are allowed in bitwise_invert')
    return Array._new(np.invert(x._array))