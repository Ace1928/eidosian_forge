import numpy as np
from scipy import stats, optimize, special
def fitquantilesgmm(distfn, x, start=None, pquant=None, frozen=None):
    if pquant is None:
        pquant = np.array([0.01, 0.05, 0.1, 0.4, 0.6, 0.9, 0.95, 0.99])
    if start is None:
        if hasattr(distfn, '_fitstart'):
            start = distfn._fitstart(x)
        else:
            start = [1] * distfn.numargs + [0.0, 1.0]
    xqs = [stats.scoreatpercentile(x, p) for p in pquant * 100]
    mom2s = None
    parest = optimize.fmin(lambda params: np.sum(momentcondquant(distfn, params, mom2s, (pquant, xqs), shape=None) ** 2), start)
    return parest