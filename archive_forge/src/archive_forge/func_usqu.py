from statsmodels.compat.python import lmap
import numpy as np
from scipy.stats import distributions
from statsmodels.tools.decorators import cache_readonly
from scipy.special import kolmogorov as ksprob
@cache_readonly
def usqu(self):
    nobs = self.nobs
    cdfvals = self.cdfvals
    usqu = self.wsqu - nobs * (cdfvals.mean() - 0.5) ** 2
    return usqu