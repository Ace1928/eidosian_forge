from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
def gmmobjective_cu(self, params, weights_method='cov', wargs=()):
    """
        objective function for continuously updating  GMM minimization

        Parameters
        ----------
        params : ndarray
            parameter values at which objective is evaluated

        Returns
        -------
        jval : float
            value of objective function

        """
    moms = self.momcond(params)
    inv_weights = self.calc_weightmatrix(moms, weights_method=weights_method, wargs=wargs)
    weights = np.linalg.pinv(inv_weights)
    self._weights_cu = weights
    return np.dot(np.dot(moms.mean(0), weights), moms.mean(0))