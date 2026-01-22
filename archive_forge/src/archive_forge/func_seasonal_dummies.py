from statsmodels.compat.pandas import frequencies
from statsmodels.compat.python import asbytes
from statsmodels.tools.validation import array_like, int_like
import numpy as np
import pandas as pd
from scipy import stats, linalg
import statsmodels.tsa.tsatools as tsa
def seasonal_dummies(n_seasons, len_endog, first_period=0, centered=False):
    """

    Parameters
    ----------
    n_seasons : int >= 0
        Number of seasons (e.g. 12 for monthly data and 4 for quarterly data).
    len_endog : int >= 0
        Total number of observations.
    first_period : int, default: 0
        Season of the first observation. As an example, suppose we have monthly
        data and the first observation is in March (third month of the year).
        In this case we pass 2 as first_period. (0 for the first season,
        1 for the second, ..., n_seasons-1 for the last season).
        An integer greater than n_seasons-1 are treated in the same way as the
        integer modulo n_seasons.
    centered : bool, default: False
        If True, center (demean) the dummy variables. That is useful in order
        to get seasonal dummies that are orthogonal to the vector of constant
        dummy variables (a vector of ones).

    Returns
    -------
    seasonal_dummies : ndarray (len_endog x n_seasons-1)
    """
    if n_seasons == 0:
        return np.empty((len_endog, 0))
    if n_seasons > 0:
        season_exog = np.zeros((len_endog, n_seasons - 1))
        for i in range(n_seasons - 1):
            season_exog[(i - first_period) % n_seasons::n_seasons, i] = 1
        if centered:
            season_exog -= 1 / n_seasons
        return season_exog