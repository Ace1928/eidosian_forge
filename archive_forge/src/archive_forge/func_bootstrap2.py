from statsmodels.compat.python import lmap
import numpy as np
from scipy.stats import distributions
from statsmodels.tools.decorators import cache_readonly
from scipy.special import kolmogorov as ksprob
def bootstrap2(value, distr, args=(), nobs=200, nrep=100):
    """Monte Carlo (or parametric bootstrap) p-values for gof

    currently hardcoded for A^2 only

    non vectorized, loops over all parametric bootstrap replications and calculates
    and returns specific p-value,

    rename function to less generic

    """
    count = 0
    for irep in range(nrep):
        rvs = distr.rvs(args, **{'size': nobs})
        params = distr.fit_vec(rvs)
        cdfvals = np.sort(distr.cdf(rvs, params))
        stat = asquare(cdfvals, axis=0)
        count += stat >= value
    return count * 1.0 / nrep