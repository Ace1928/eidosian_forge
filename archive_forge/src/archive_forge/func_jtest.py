from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
def jtest(self):
    """overidentification test

        I guess this is missing a division by nobs,
        what's the normalization in jval ?
        """
    jstat = self.jval
    nparams = self.params.size
    df = self.model.nmoms - nparams
    return (jstat, stats.chi2.sf(jstat, df), df)