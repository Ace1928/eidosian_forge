import numpy as np
from statsmodels.tools.grouputils import combine_indices, group_sums
from statsmodels.stats.moment_helpers import se_cov
def cov_hc3(results):
    """
    See statsmodels.RegressionResults
    """
    h = np.diag(np.dot(results.model.exog, np.dot(results.normalized_cov_params, results.model.exog.T)))
    het_scale = (results.resid / (1 - h)) ** 2
    cov_hc3_ = _HCCM(results, het_scale)
    return cov_hc3_