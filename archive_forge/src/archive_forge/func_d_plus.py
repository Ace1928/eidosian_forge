from statsmodels.compat.python import lmap
import numpy as np
from scipy.stats import distributions
from statsmodels.tools.decorators import cache_readonly
from scipy.special import kolmogorov as ksprob
@cache_readonly
def d_plus(self):
    nobs = self.nobs
    cdfvals = self.cdfvals
    return (np.arange(1.0, nobs + 1) / nobs - cdfvals).max()