from __future__ import annotations
import io
import sys
import typing as ty
import warnings
from functools import reduce
from operator import getitem, mul
from os.path import exists, splitext
import numpy as np
from ._compression import COMPRESSED_FILE_LIKES
from .casting import OK_FLOATS, shared_range
from .externals.oset import OrderedSet
def better_float_of(first: npt.DTypeLike, second: npt.DTypeLike, default: type[np.floating]=np.float32) -> type[np.floating]:
    """Return more capable float type of `first` and `second`

    Return `default` if neither of `first` or `second` is a float

    Parameters
    ----------
    first : numpy type specifier
        Any valid input to `np.dtype()``
    second : numpy type specifier
        Any valid input to `np.dtype()``
    default : numpy type specifier, optional
        Any valid input to `np.dtype()``

    Returns
    -------
    better_type : numpy type
        More capable of `first` or `second` if both are floats; if only one is
        a float return that, otherwise return `default`.

    Examples
    --------
    >>> better_float_of(np.float32, np.float64) is np.float64
    True
    >>> better_float_of(np.float32, 'i4') is np.float32
    True
    >>> better_float_of('i2', 'u4') is np.float32
    True
    >>> better_float_of('i2', 'u4', np.float64) is np.float64
    True
    """
    first = np.dtype(first)
    second = np.dtype(second)
    default = np.dtype(default).type
    if issubclass(first.type, np.floating):
        if issubclass(second.type, np.floating) and first.itemsize < second.itemsize:
            return second.type
        return first.type
    if issubclass(second.type, np.floating):
        return second.type
    return default