from statsmodels.compat.python import lmap
import numpy as np
from scipy.stats import distributions
from statsmodels.tools.decorators import cache_readonly
from scipy.special import kolmogorov as ksprob
@cache_readonly
def wsqu(self):
    """Cramer von Mises"""
    nobs = self.nobs
    cdfvals = self.cdfvals
    wsqu = ((cdfvals - (2.0 * np.arange(1.0, nobs + 1) - 1) / nobs / 2.0) ** 2).sum() + 1.0 / nobs / 12.0
    return wsqu