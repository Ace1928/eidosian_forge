from __future__ import annotations
import typing
from contextlib import suppress
from warnings import warn
import numpy as np
import pandas as pd
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.evaluation import after_stat
from .stat import stat
def nrd0(x: FloatArrayLike) -> float:
    """
    Port of R stats::bw.nrd0

    This is equivalent to statsmodels silverman when x has more than
    1 unique value. It can never give a zero bandwidth.

    Parameters
    ----------
    x : array_like
        Values whose density is to be estimated

    Returns
    -------
    out : float
        Bandwidth of x
    """
    from scipy.stats import iqr
    n = len(x)
    if n < 1:
        raise ValueError('Need at leat 2 data points to compute the nrd0 bandwidth.')
    std: float = np.std(x, ddof=1)
    std_estimate: float = iqr(x) / 1.349
    low_std = min(std, std_estimate)
    if low_std == 0:
        low_std = std_estimate or np.abs(np.asarray(x)[0]) or 1
    return 0.9 * low_std * n ** (-0.2)