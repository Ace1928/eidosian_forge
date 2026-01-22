import numpy as np
from numpy.linalg import eigvals, inv, solve, matrix_rank, pinv, svd
from scipy import stats
import pandas as pd
from patsy import DesignInfo
from statsmodels.compat.pandas import Substitution
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
@Substitution(hypotheses_doc=_hypotheses_doc)
def mv_test(self, hypotheses=None, skip_intercept_test=False):
    """
        Linear hypotheses testing

        Parameters
        ----------
        %(hypotheses_doc)s
        skip_intercept_test : bool
            If true, then testing the intercept is skipped, the model is not
            changed.
            Note: If a term has a numerically insignificant effect, then
            an exception because of emtpy arrays may be raised. This can
            happen for the intercept if the data has been demeaned.

        Returns
        -------
        results: _MultivariateOLSResults

        Notes
        -----
        Tests hypotheses of the form

            L * params * M = C

        where `params` is the regression coefficient matrix for the
        linear model y = x * params, `L` is the contrast matrix, `M` is the
        dependent variable transform matrix and C is the constant matrix.
        """
    k_xvar = len(self.exog_names)
    if hypotheses is None:
        if self.design_info is not None:
            terms = self.design_info.term_name_slices
            hypotheses = []
            for key in terms:
                if skip_intercept_test and key == 'Intercept':
                    continue
                L_contrast = np.eye(k_xvar)[terms[key], :]
                hypotheses.append([key, L_contrast, None])
        else:
            hypotheses = []
            for i in range(k_xvar):
                name = 'x%d' % i
                L = np.zeros([1, k_xvar])
                L[i] = 1
                hypotheses.append([name, L, None])
    results = _multivariate_ols_test(hypotheses, self._fittedmod, self.exog_names, self.endog_names)
    return MultivariateTestResults(results, self.endog_names, self.exog_names)