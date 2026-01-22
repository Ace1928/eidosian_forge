from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
def fitstart(self):
    distfn = self.distfn
    if hasattr(distfn, '_fitstart'):
        start = distfn._fitstart(self.endog)
    else:
        start = [1] * distfn.numargs + [0.0, 1.0]
    return np.asarray(start)