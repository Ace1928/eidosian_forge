from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
def fititer(self, start, maxiter=2, start_invweights=None, weights_method='cov', wargs=(), optim_method='bfgs', optim_args=None):
    """iterative estimation with updating of optimal weighting matrix

        stopping criteria are maxiter or change in parameter estimate less
        than self.epsilon_iter, with default 1e-6.

        Parameters
        ----------
        start : ndarray
            starting value for parameters
        maxiter : int
            maximum number of iterations
        start_weights : array (nmoms, nmoms)
            initial weighting matrix; if None, then the identity matrix
            is used
        weights_method : {'cov', ...}
            method to use to estimate the optimal weighting matrix,
            see calc_weightmatrix for details

        Returns
        -------
        params : ndarray
            estimated parameters
        weights : ndarray
            optimal weighting matrix calculated with final parameter
            estimates

        Notes
        -----




        """
    self.history = []
    momcond = self.momcond
    if start_invweights is None:
        w = self.start_weights(inv=True)
    else:
        w = start_invweights
    winv_new = w
    for it in range(maxiter):
        winv = winv_new
        w = np.linalg.pinv(winv)
        resgmm = self.fitgmm(start, weights=w, optim_method=optim_method, optim_args=optim_args)
        moms = momcond(resgmm)
        winv_new = self.calc_weightmatrix(moms, weights_method=weights_method, wargs=wargs, params=resgmm)
        if it > 2 and maxabs(resgmm - start) < self.epsilon_iter:
            break
        start = resgmm
    return (resgmm, w)