from __future__ import annotations
import datetime
import sys
import typing
from copy import copy
from typing import overload
import numpy as np
import pandas as pd
from .utils import get_null_value, is_vector
def rescale_max(x: FloatArrayLike, to: TupleFloat2=(0, 1), _from: Optional[TupleFloat2]=None) -> NDArrayFloat:
    """
    Rescale numeric vector to have specified maximum.

    Parameters
    ----------
    x : array_like
        1D vector of values to manipulate.
    to : tuple
        output range (numeric vector of length two)
    _from : tuple
        input range (numeric vector of length two).
        If not given, is calculated from the range of x.
        Only the 2nd (max) element is essential to the
        output.

    Returns
    -------
    out : array_like
        Rescaled values

    Examples
    --------
    >>> x = np.array([0, 2, 4, 6, 8, 10])
    >>> rescale_max(x, (0, 3))
    array([0. , 0.6, 1.2, 1.8, 2.4, 3. ])

    Only the 2nd (max) element of the parameters ``to``
    and ``_from`` are essential to the output.

    >>> rescale_max(x, (1, 3))
    array([0. , 0.6, 1.2, 1.8, 2.4, 3. ])
    >>> rescale_max(x, (0, 20))
    array([ 0.,  4.,  8., 12., 16., 20.])

    If :python:`max(x) < _from[1]` then values will be
    scaled beyond the requested maximum (:python:`to[1]`).

    >>> rescale_max(x, to=(1, 3), _from=(-1, 6))
    array([0., 1., 2., 3., 4., 5.])

    If the values are the same, they taken on the requested maximum.
    This includes an array of all zeros.

    >>> rescale_max(np.array([5, 5, 5]))
    array([1., 1., 1.])
    >>> rescale_max(np.array([0, 0, 0]))
    array([1, 1, 1])
    """
    x = np.asarray(x)
    if _from is None:
        _from = (np.min(x), np.max(x))
        assert _from is not None
    if np.any(x < 0):
        out = rescale(x, (0, to[1]), _from)
    elif np.all(x == 0) and _from[1] == 0:
        out = np.repeat(to[1], len(x))
    else:
        out = x / _from[1] * to[1]
    return out