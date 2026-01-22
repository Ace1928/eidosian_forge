import numpy as np
from scipy import optimize
from statsmodels.base.model import Model
def fit_random(self, ntries=10, rvs_generator=None, nparams=None):
    """fit with random starting values

        this could be replaced with a global fitter

        """
    if nparams is None:
        nparams = self.nparams
    if rvs_generator is None:
        rvs = np.random.uniform(low=-10, high=10, size=(ntries, nparams))
    else:
        rvs = rvs_generator(size=(ntries, nparams))
    results = np.array([np.r_[self.fit_minimal(rv), rv] for rv in rvs])
    return results