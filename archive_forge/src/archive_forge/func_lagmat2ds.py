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
def lagmat2ds(x, maxlag0, maxlagex=None, dropex=0, trim='forward', use_pandas=False):
    """
    Generate lagmatrix for 2d array, columns arranged by variables.

    Parameters
    ----------
    x : array_like
        Data, 2d. Observations in rows and variables in columns.
    maxlag0 : int
        The first variable all lags from zero to maxlag are included.
    maxlagex : {None, int}
        The max lag for all other variables all lags from zero to maxlag are
        included.
    dropex : int
        Exclude first dropex lags from other variables. For all variables,
        except the first, lags from dropex to maxlagex are included.
    trim : str
        The trimming method to use.

        * 'forward' : trim invalid observations in front.
        * 'backward' : trim invalid initial observations.
        * 'both' : trim invalid observations on both sides.
        * 'none' : no trimming of observations.
    use_pandas : bool
        If true, returns a DataFrame when the input is a pandas
        Series or DataFrame.  If false, return numpy ndarrays.

    Returns
    -------
    ndarray
        The array with lagged observations, columns ordered by variable.

    Notes
    -----
    Inefficient implementation for unequal lags, implemented for convenience.
    """
    maxlag0 = int_like(maxlag0, 'maxlag0')
    maxlagex = int_like(maxlagex, 'maxlagex', optional=True)
    trim = string_like(trim, 'trim', optional=True, options=('forward', 'backward', 'both', 'none'))
    if maxlagex is None:
        maxlagex = maxlag0
    maxlag = max(maxlag0, maxlagex)
    is_pandas = _is_using_pandas(x, None)
    if x.ndim == 1:
        if is_pandas:
            x = pd.DataFrame(x)
        else:
            x = x[:, None]
    elif x.ndim == 0 or x.ndim > 2:
        raise ValueError('Only supports 1 and 2-dimensional data.')
    nobs, nvar = x.shape
    if is_pandas and use_pandas:
        lags = lagmat(x.iloc[:, 0], maxlag, trim=trim, original='in', use_pandas=True)
        lagsli = [lags.iloc[:, :maxlag0 + 1]]
        for k in range(1, nvar):
            lags = lagmat(x.iloc[:, k], maxlag, trim=trim, original='in', use_pandas=True)
            lagsli.append(lags.iloc[:, dropex:maxlagex + 1])
        return pd.concat(lagsli, axis=1)
    elif is_pandas:
        x = np.asanyarray(x)
    lagsli = [lagmat(x[:, 0], maxlag, trim=trim, original='in')[:, :maxlag0 + 1]]
    for k in range(1, nvar):
        lagsli.append(lagmat(x[:, k], maxlag, trim=trim, original='in')[:, dropex:maxlagex + 1])
    return np.column_stack(lagsli)