from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
def spec_hausman(self, dof=None):
    """Hausman's specification test

        See Also
        --------
        spec_hausman : generic function for Hausman's specification test

        """
    endog, exog = (self.model.endog, self.model.exog)
    resols = OLS(endog, exog).fit()
    normalized_cov_params_ols = resols.model.normalized_cov_params
    se2 = resols.ssr / len(endog)
    params_diff = self.params - resols.params
    cov_diff = np.linalg.pinv(self.model.xhatprod) - normalized_cov_params_ols
    if not dof:
        dof = np.linalg.matrix_rank(cov_diff)
    cov_diffpinv = np.linalg.pinv(cov_diff)
    H = np.dot(params_diff, np.dot(cov_diffpinv, params_diff)) / se2
    pval = stats.chi2.sf(H, dof)
    return (H, pval, dof)