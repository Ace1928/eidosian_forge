import numpy as np
import pandas as pd
import scipy.linalg
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.validation import array_like
def recipr(x):
    """
    Reciprocal of an array with entries less than or equal to 0 set to 0.

    Parameters
    ----------
    x : array_like
        The input array.

    Returns
    -------
    ndarray
        The array with 0-filled reciprocals.
    """
    x = np.asarray(x)
    out = np.zeros_like(x, dtype=np.float64)
    nans = np.isnan(x.flat)
    pos = ~nans
    pos[pos] = pos[pos] & (x.flat[pos] > 0)
    out.flat[pos] = 1.0 / x.flat[pos]
    out.flat[nans] = np.nan
    return out