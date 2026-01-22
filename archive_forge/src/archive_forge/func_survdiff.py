import numpy as np
import pandas as pd
from scipy.stats.distributions import chi2, norm
from statsmodels.graphics import utils
def survdiff(time, status, group, weight_type=None, strata=None, entry=None, **kwargs):
    """
    Test for the equality of two survival distributions.

    Parameters
    ----------
    time : array_like
        The event or censoring times.
    status : array_like
        The censoring status variable, status=1 indicates that the
        event occurred, status=0 indicates that the observation was
        censored.
    group : array_like
        Indicators of the two groups
    weight_type : str
        The following weight types are implemented:
            None (default) : logrank test
            fh : Fleming-Harrington, weights by S^(fh_p),
                 requires exponent fh_p to be provided as keyword
                 argument; the weights are derived from S defined at
                 the previous event time, and the first weight is
                 always 1.
            gb : Gehan-Breslow, weights by the number at risk
            tw : Tarone-Ware, weights by the square root of the number
                 at risk
    strata : array_like
        Optional stratum indicators for a stratified test
    entry : array_like
        Entry times to handle left truncation. The subject is not in
        the risk set on or before the entry time.

    Returns
    -------
    chisq : The chi-square (1 degree of freedom) distributed test
            statistic value
    pvalue : The p-value for the chi^2 test
    """
    time = np.asarray(time)
    status = np.asarray(status)
    group = np.asarray(group)
    gr = np.unique(group)
    if strata is None:
        obs, var = _survdiff(time, status, group, weight_type, gr, entry, **kwargs)
    else:
        strata = np.asarray(strata)
        stu = np.unique(strata)
        obs, var = (0.0, 0.0)
        for st in stu:
            ii = strata == st
            obs1, var1 = _survdiff(time[ii], status[ii], group[ii], weight_type, gr, entry, **kwargs)
            obs += obs1
            var += var1
    chisq = obs.dot(np.linalg.solve(var, obs))
    pvalue = 1 - chi2.cdf(chisq, len(gr) - 1)
    return (chisq, pvalue)