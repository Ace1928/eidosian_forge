import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def log_riskratio(self):
    """
        Returns the log of the risk ratio.
        """
    return np.log(self.riskratio)