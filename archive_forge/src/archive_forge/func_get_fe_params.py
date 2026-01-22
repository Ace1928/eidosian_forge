import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.stats.distributions import norm
from statsmodels.base._penalties import Penalty
import statsmodels.base.model as base
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
def get_fe_params(self, cov_re, vcomp, tol=1e-10):
    """
        Use GLS to update the fixed effects parameter estimates.

        Parameters
        ----------
        cov_re : array_like (2d)
            The covariance matrix of the random effects.
        vcomp : array_like (1d)
            The variance components.
        tol : float
            A tolerance parameter to determine when covariances
            are singular.

        Returns
        -------
        params : ndarray
            The GLS estimates of the fixed effects parameters.
        singular : bool
            True if the covariance is singular
        """
    if self.k_fe == 0:
        return (np.array([]), False)
    sing = False
    if self.k_re == 0:
        cov_re_inv = np.empty((0, 0))
    else:
        w, v = np.linalg.eigh(cov_re)
        if w.min() < tol:
            sing = True
            ii = np.flatnonzero(w >= tol)
            if len(ii) == 0:
                cov_re_inv = np.zeros_like(cov_re)
            else:
                vi = v[:, ii]
                wi = w[ii]
                cov_re_inv = np.dot(vi / wi, vi.T)
        else:
            cov_re_inv = np.linalg.inv(cov_re)
    if not hasattr(self, '_endex_li'):
        self._endex_li = []
        for group_ix, _ in enumerate(self.group_labels):
            mat = np.concatenate((self.exog_li[group_ix], self.endog_li[group_ix][:, None]), axis=1)
            self._endex_li.append(mat)
    xtxy = 0.0
    for group_ix, group in enumerate(self.group_labels):
        vc_var = self._expand_vcomp(vcomp, group_ix)
        if vc_var.size > 0:
            if vc_var.min() < tol:
                sing = True
                ii = np.flatnonzero(vc_var >= tol)
                vc_vari = np.zeros_like(vc_var)
                vc_vari[ii] = 1 / vc_var[ii]
            else:
                vc_vari = 1 / vc_var
        else:
            vc_vari = np.empty(0)
        exog = self.exog_li[group_ix]
        ex_r, ex2_r = (self._aex_r[group_ix], self._aex_r2[group_ix])
        solver = _smw_solver(1.0, ex_r, ex2_r, cov_re_inv, vc_vari)
        u = solver(self._endex_li[group_ix])
        xtxy += np.dot(exog.T, u)
    if sing:
        fe_params = np.dot(np.linalg.pinv(xtxy[:, 0:-1]), xtxy[:, -1])
    else:
        fe_params = np.linalg.solve(xtxy[:, 0:-1], xtxy[:, -1])
    return (fe_params, sing)