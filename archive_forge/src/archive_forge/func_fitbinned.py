import numpy as np
from scipy import stats, optimize, special
def fitbinned(distfn, freq, binedges, start, fixed=None):
    """estimate parameters of distribution function for binned data using MLE

    Parameters
    ----------
    distfn : distribution instance
        needs to have cdf method, as in scipy.stats
    freq : ndarray, 1d
        frequency count, e.g. obtained by histogram
    binedges : ndarray, 1d
        binedges including lower and upper bound
    start : tuple or array_like ?
        starting values, needs to have correct length

    Returns
    -------
    paramest : ndarray
        estimated parameters

    Notes
    -----
    todo: add fixed parameter option

    added factorial

    """
    if fixed is not None:
        raise NotImplementedError
    nobs = np.sum(freq)
    lnnobsfact = special.gammaln(nobs + 1)

    def nloglike(params):
        """negative loglikelihood function of binned data

        corresponds to multinomial
        """
        prob = np.diff(distfn.cdf(binedges, *params))
        return -(lnnobsfact + np.sum(freq * np.log(prob) - special.gammaln(freq + 1)))
    return optimize.fmin(nloglike, start)