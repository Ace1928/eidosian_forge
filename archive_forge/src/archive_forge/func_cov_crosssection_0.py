import numpy as np
from statsmodels.tools.grouputils import combine_indices, group_sums
from statsmodels.stats.moment_helpers import se_cov
def cov_crosssection_0(results, group):
    """this one is still wrong, use cov_cluster instead"""
    scale = S_crosssection(results.resid[:, None], group)
    scale = np.squeeze(scale)
    cov = _HCCM1(results, scale)
    return cov