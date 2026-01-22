import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def logodds_pooled(self):
    """
        Returns the logarithm of the pooled odds ratio.

        See oddsratio_pooled for more information.
        """
    return np.log(self.oddsratio_pooled)