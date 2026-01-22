import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
def log_riskratio_pvalue(self, null=0):
    """
        p-value for a hypothesis test about the log risk ratio.

        Parameters
        ----------
        null : float
            The null value of the log risk ratio.
        """
    zscore = (self.log_riskratio - null) / self.log_riskratio_se
    pvalue = 2 * stats.norm.cdf(-np.abs(zscore))
    return pvalue