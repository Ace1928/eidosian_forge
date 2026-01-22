import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
def oddsratio_pooled_confint(self, alpha=0.05, method='normal'):
    """
        A confidence interval for the pooled odds ratio.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the nominal coverage probability of the
            interval.
        method : str
            The method for producing the confidence interval.  Currently
            must be 'normal' which uses the normal approximation.

        Returns
        -------
        lcb : float
            The lower confidence limit.
        ucb : float
            The upper confidence limit.
        """
    lcb, ucb = self.logodds_pooled_confint(alpha, method=method)
    lcb = np.exp(lcb)
    ucb = np.exp(ucb)
    return (lcb, ucb)