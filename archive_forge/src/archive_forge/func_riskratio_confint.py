import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
def riskratio_confint(self, alpha=0.05, method='normal'):
    """
        A confidence interval for the risk ratio.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the nominal coverage probability of the
            confidence interval.
        method : str
            The method for producing the confidence interval.  Currently
            must be 'normal' which uses the normal approximation.
        """
    lcb, ucb = self.log_riskratio_confint(alpha, method=method)
    return (np.exp(lcb), np.exp(ucb))