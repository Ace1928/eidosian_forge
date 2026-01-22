from a common formula are constrained to have the same standard
import numpy as np
from scipy.optimize import minimize
from scipy import sparse
import statsmodels.base.model as base
from statsmodels.iolib import summary2
from statsmodels.genmod import families
import pandas as pd
import warnings
import patsy
def fit_map(self, method='BFGS', minim_opts=None, scale_fe=False):
    """
        Construct the Laplace approximation to the posterior distribution.

        Parameters
        ----------
        method : str
            Optimization method for finding the posterior mode.
        minim_opts : dict
            Options passed to scipy.minimize.
        scale_fe : bool
            If True, the columns of the fixed effects design matrix
            are centered and scaled to unit variance before fitting
            the model.  The results are back-transformed so that the
            results are presented on the original scale.

        Returns
        -------
        BayesMixedGLMResults instance.
        """
    if scale_fe:
        mn = self.exog.mean(0)
        sc = self.exog.std(0)
        self._exog_save = self.exog
        self.exog = self.exog.copy()
        ixs = np.flatnonzero(sc > 1e-08)
        self.exog[:, ixs] -= mn[ixs]
        self.exog[:, ixs] /= sc[ixs]

    def fun(params):
        return -self.logposterior(params)

    def grad(params):
        return -self.logposterior_grad(params)
    start = self._get_start()
    r = minimize(fun, start, method=method, jac=grad, options=minim_opts)
    if not r.success:
        msg = 'Laplace fitting did not converge, |gradient|=%.6f' % np.sqrt(np.sum(r.jac ** 2))
        warnings.warn(msg)
    from statsmodels.tools.numdiff import approx_fprime
    hess = approx_fprime(r.x, grad)
    cov = np.linalg.inv(hess)
    params = r.x
    if scale_fe:
        self.exog = self._exog_save
        del self._exog_save
        params[ixs] /= sc[ixs]
        cov[ixs, :][:, ixs] /= np.outer(sc[ixs], sc[ixs])
    return BayesMixedGLMResults(self, params, cov, optim_retvals=r)