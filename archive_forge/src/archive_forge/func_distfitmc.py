from statsmodels.compat.python import lmap
import numpy as np
from scipy import stats, optimize, integrate
def distfitmc(sample, distr, nrepl=100, distkwds={}):
    """run Monte Carlo for estimation of distribution parameters

    hard coded: only one shape parameter is allowed and estimated,
        loc=0 and scale=1 are fixed in the estimation

    Parameters
    ----------
    sample : ndarray
        original sample data, in Monte Carlo only used to get nobs,
    distr : distribution instance with fit_fr method
    nrepl : int
        number of Monte Carlo replications

    Returns
    -------
    res : array (nrepl,)
        parameter estimates for all Monte Carlo replications

    """
    arg = distkwds.pop('arg')
    nobs = len(sample)
    res = np.zeros(nrepl)
    for ii in range(nrepl):
        x = distr.rvs(arg, size=nobs, **distkwds)
        res[ii] = distr.fit_fr(x, frozen=[np.nan, 0.0, 1.0])
    return res