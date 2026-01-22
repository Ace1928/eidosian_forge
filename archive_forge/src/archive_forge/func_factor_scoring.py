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
def factor_scoring(self, endog=None, method='bartlett', transform=True):
    """
        factor scoring: compute factors for endog

        If endog was not provided when creating the factor class, then
        a standarized endog needs to be provided here.

        Parameters
        ----------
        method : 'bartlett' or 'regression'
            Method to use for factor scoring.
            'regression' can be abbreviated to `reg`
        transform : bool
            If transform is true and endog is provided, then it will be
            standardized using mean and scale of original data, which has to
            be available in this case.
            If transform is False, then a provided endog will be used unchanged.
            The original endog in the Factor class will
            always be standardized if endog is None, independently of `transform`.

        Returns
        -------
        factor_score : ndarray
            estimated factors using scoring matrix s and standarized endog ys
            ``f = ys dot s``

        Notes
        -----
        Status: transform option is experimental and might change.

        See Also
        --------
        statsmodels.multivariate.factor.FactorResults.factor_score_params
        """
    if transform is False and endog is not None:
        endog = np.asarray(endog)
    else:
        if self.model.endog is not None:
            m = self.model.endog.mean(0)
            s = self.model.endog.std(ddof=1, axis=0)
            if endog is None:
                endog = self.model.endog
            else:
                endog = np.asarray(endog)
        else:
            raise ValueError('If transform is True, then `endog` needs ' + 'to be available in the Factor instance.')
        endog = (endog - m) / s
    s_mat = self.factor_score_params(method=method)
    factors = endog.dot(s_mat)
    return factors