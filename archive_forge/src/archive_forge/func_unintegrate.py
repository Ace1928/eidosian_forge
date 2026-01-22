from __future__ import annotations
from statsmodels.compat.python import lrange
import warnings
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from typing import Literal
from statsmodels.tools.data import _is_recarray, _is_using_pandas
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.typing import NDArray
from statsmodels.tools.validation import (
def unintegrate(x, levels):
    """
    After taking n-differences of a series, return the original series

    Parameters
    ----------
    x : array_like
        The n-th differenced series
    levels : list
        A list of the first-value in each differenced series, for
        [first-difference, second-difference, ..., n-th difference]

    Returns
    -------
    y : array_like
        The original series de-differenced

    Examples
    --------
    >>> x = np.array([1, 3, 9., 19, 8.])
    >>> levels = unintegrate_levels(x, 2)
    >>> levels
    array([ 1.,  2.])
    >>> unintegrate(np.diff(x, 2), levels)
    array([  1.,   3.,   9.,  19.,   8.])
    """
    levels = list(levels)[:]
    if len(levels) > 1:
        x0 = levels.pop(-1)
        return unintegrate(np.cumsum(np.r_[x0, x]), levels)
    x0 = levels[0]
    return np.cumsum(np.r_[x0, x])