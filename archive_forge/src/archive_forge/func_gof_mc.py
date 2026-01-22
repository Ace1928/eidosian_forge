from statsmodels.compat.python import lmap
import numpy as np
from scipy.stats import distributions
from statsmodels.tools.decorators import cache_readonly
from scipy.special import kolmogorov as ksprob
def gof_mc(randfn, distr, nobs=100):
    from collections import defaultdict
    results = defaultdict(list)
    for i in range(1000):
        rvs = randfn(nobs)
        goft = GOF(rvs, distr)
        for ti in all_gofs:
            results[ti].append(goft.get_test(ti, 'stephens70upp')[0][1])
    resarr = np.array([results[ti] for ti in all_gofs])
    print('         ', '      '.join(all_gofs))
    print('at 0.01:', (resarr < 0.01).mean(1))
    print('at 0.05:', (resarr < 0.05).mean(1))
    print('at 0.10:', (resarr < 0.1).mean(1))