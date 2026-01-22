from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
def calc_cov_params(self, moms, gradmoms, weights=None, use_weights=False, has_optimal_weights=True, weights_method='cov', wargs=()):
    """calculate covariance of parameter estimates

        not all options tried out yet

        If weights matrix is given, then the formula use to calculate cov_params
        depends on whether has_optimal_weights is true.
        If no weights are given, then the weight matrix is calculated with
        the given method, and has_optimal_weights is assumed to be true.

        (API Note: The latter assumption could be changed if we allow for
        has_optimal_weights=None.)

        """
    nobs = moms.shape[0]
    if weights is None:
        weights = self.weights
    else:
        pass
    if use_weights:
        omegahat = weights
    else:
        omegahat = self.model.calc_weightmatrix(moms, weights_method=weights_method, wargs=wargs, params=self.params)
    if has_optimal_weights:
        cov = np.linalg.inv(np.dot(gradmoms.T, np.dot(np.linalg.inv(omegahat), gradmoms)))
    else:
        gw = np.dot(gradmoms.T, weights)
        gwginv = np.linalg.inv(np.dot(gw, gradmoms))
        cov = np.dot(np.dot(gwginv, np.dot(np.dot(gw, omegahat), gw.T)), gwginv)
    return cov / nobs