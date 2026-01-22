from statsmodels.compat.python import lmap
import numpy as np
from scipy.stats import distributions
from statsmodels.tools.decorators import cache_readonly
from scipy.special import kolmogorov as ksprob
@cache_readonly
def d_minus(self):
    nobs = self.nobs
    cdfvals = self.cdfvals
    return (cdfvals - np.arange(0.0, nobs) / nobs).max()