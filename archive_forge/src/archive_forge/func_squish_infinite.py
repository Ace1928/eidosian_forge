from __future__ import annotations
import datetime
import sys
import typing
from copy import copy
from typing import overload
import numpy as np
import pandas as pd
from .utils import get_null_value, is_vector
def squish_infinite(x: FloatArrayLike, range: TupleFloat2=(0, 1)) -> NDArrayFloat:
    """
    Truncate infinite values to a range.

    Parameters
    ----------
    x : array_like
        Values that should have infinities squished.
    range : tuple
        The range onto which to squish the infinites.
        Must be of size 2.

    Returns
    -------
    out : array_like
        Values with infinites squished.

    Examples
    --------
    >>> arr1 = np.array([0, .5, .25, np.inf, .44])
    >>> arr2 = np.array([0, -np.inf, .5, .25, np.inf])
    >>> list(squish_infinite(arr1))
    [0.0, 0.5, 0.25, 1.0, 0.44]
    >>> list(squish_infinite(arr2, (-10, 9)))
    [0.0, -10.0, 0.5, 0.25, 9.0]
    """
    _x = np.asarray(x)
    _x[np.isneginf(_x)] = range[0]
    _x[np.isposinf(_x)] = range[1]
    return _x