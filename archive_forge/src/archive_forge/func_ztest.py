import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
def ztest(self):
    """statistic, p-value and degrees of freedom of separate moment test

        currently two sided test only

        TODO: This can use generic ztest/ttest features and return
        ContrastResults
        """
    diff = self.moments_constraint
    bse = np.sqrt(np.diag(self.cov_mom_constraints))
    stat = diff / bse
    pval = stats.norm.sf(np.abs(stat)) * 2
    return (stat, pval)