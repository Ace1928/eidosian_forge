from __future__ import annotations
import datetime
import sys
import typing
from copy import copy
from typing import overload
import numpy as np
import pandas as pd
from .utils import get_null_value, is_vector
def censor(x: NDArrayFloat | Sequence[float] | FloatSeries, range: TupleFloat2=(0, 1), only_finite: bool=True) -> NDArrayFloat | FloatSeries:
    """
    Convert any values outside of range to a **NULL** type object.

    Parameters
    ----------
    x : array_like
        Values to manipulate
    range : tuple
        (min, max) giving desired output range
    only_finite : bool
        If True (the default), will only modify
        finite values.

    Returns
    -------
    x : array_like
        Censored array

    Examples
    --------
    >>> a = np.array([1, 2, np.inf, 3, 4, -np.inf, 5])
    >>> list(censor(a, (0, 10)))
    [1.0, 2.0, inf, 3.0, 4.0, -inf, 5.0]
    >>> list(censor(a, (0, 10), False))
    [1.0, 2.0, nan, 3.0, 4.0, nan, 5.0]
    >>> list(censor(a, (2, 4)))
    [nan, 2.0, inf, 3.0, 4.0, -inf, nan]

    Notes
    -----
    All values in ``x`` should be of the same type. ``only_finite`` parameter
    is not considered for Datetime and Timedelta types.

    The **NULL** type object depends on the type of values in **x**.

    - :class:`float` - :py:`float('nan')`
    - :class:`int` - :py:`float('nan')`
    - :class:`datetime.datetime` : :py:`np.datetime64(NaT)`
    - :class:`datetime.timedelta` : :py:`np.timedelta64(NaT)`

    """
    if not len(x):
        return np.array([])
    if not is_vector(x):
        x = np.asarray(x)
    null = get_null_value(x)
    if only_finite:
        try:
            finite = np.isfinite(x)
        except TypeError:
            finite = np.repeat(True, len(x))
    else:
        finite = np.repeat(True, len(x))
    with np.errstate(invalid='ignore'):
        outside = (x < range[0]) | (x > range[1])
    bool_idx = finite & outside
    res = copy(x)
    if bool_idx.any():
        if res.dtype.kind == 'i':
            res = np.asarray(res, dtype=float)
        res[bool_idx] = null
    return res