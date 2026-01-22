import warnings
import numpy as np
from numpy.linalg import eigh, inv, norm, matrix_rank
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
from statsmodels.graphics.utils import _import_mpl
from .factor_rotation import rotate_factors, promax
def factor_score_params(self, method='bartlett'):
    """
        Compute factor scoring coefficient matrix

        The coefficient matrix is not cached.

        Parameters
        ----------
        method : 'bartlett' or 'regression'
            Method to use for factor scoring.
            'regression' can be abbreviated to `reg`

        Returns
        -------
        coeff_matrix : ndarray
            matrix s to compute factors f from a standardized endog ys.
            ``f = ys dot s``

        Notes
        -----
        The `regression` method follows the Stata definition.
        Method bartlett and regression are verified against Stats.
        Two unofficial methods, 'ols' and 'gls', produce similar factor scores
        but are not verified.

        See Also
        --------
        statsmodels.multivariate.factor.FactorResults.factor_scoring
        """
    L = self.loadings
    T = self.rotation_matrix.T
    uni = 1 - self.communality
    if method == 'bartlett':
        s_mat = np.linalg.inv(L.T.dot(L / uni[:, None])).dot(L.T / uni).T
    elif method.startswith('reg'):
        corr = self.model.corr
        corr_f = self._corr_factors()
        s_mat = corr_f.dot(L.T.dot(np.linalg.inv(corr))).T
    elif method == 'ols':
        corr = self.model.corr
        corr_f = self._corr_factors()
        s_mat = corr_f.dot(np.linalg.pinv(L)).T
    elif method == 'gls':
        corr = self.model.corr
        corr_f = self._corr_factors()
        s_mat = np.linalg.inv(np.linalg.inv(corr_f) + L.T.dot(L / uni[:, None]))
        s_mat = s_mat.dot(L.T / uni).T
    else:
        raise ValueError('method not available, use "bartlett ' + 'or "regression"')
    return s_mat