import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
def riskratio_pvalue(self, null=1):
    """
        p-value for a hypothesis test about the risk ratio.

        Parameters
        ----------
        null : float
            The null value of the risk ratio.
        """
    return self.log_riskratio_pvalue(np.log(null))