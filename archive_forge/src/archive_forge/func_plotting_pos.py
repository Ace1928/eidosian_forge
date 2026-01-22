from statsmodels.compat.python import lzip
import numpy as np
from scipy import stats
from statsmodels.distributions import ECDF
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import add_constant
from . import utils
def plotting_pos(nobs, a=0.0, b=None):
    """
    Generates sequence of plotting positions

    Parameters
    ----------
    nobs : int
        Number of probability points to plot
    a : float, default 0.0
        alpha parameter for the plotting position of an expected order
        statistic
    b : float, default None
        beta parameter for the plotting position of an expected order
        statistic. If None, then b is set to a.

    Returns
    -------
    ndarray
        The plotting positions

    Notes
    -----
    The plotting positions are given by (i - a)/(nobs + 1 - a - b) for i in
    range(1, nobs+1)

    See Also
    --------
    scipy.stats.mstats.plotting_positions
        Additional information on alpha and beta
    """
    b = a if b is None else b
    return (np.arange(1.0, nobs + 1) - a) / (nobs + 1 - a - b)