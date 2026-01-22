import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def log_riskratio_se(self):
    """
        Returns the standard error of the log of the risk ratio.
        """
    n = self.table.sum(1)
    p = self.table[:, 0] / n
    va = np.sum((1 - p) / (n * p))
    return np.sqrt(va)