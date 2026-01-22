from __future__ import annotations
from ._array_object import Array
from typing import NamedTuple
import cupy as np
def unique_all(x: Array, /) -> UniqueAllResult:
    """
    Array API compatible wrapper for :py:func:`np.unique <numpy.unique>`.

    See its docstring for more information.
    """
    values, indices, inverse_indices, counts = np.unique(x._array, return_counts=True, return_index=True, return_inverse=True, equal_nan=False)
    inverse_indices = inverse_indices.reshape(x.shape)
    return UniqueAllResult(Array._new(values), Array._new(indices), Array._new(inverse_indices), Array._new(counts))