import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def log_oddsratio(self):
    """
        Returns the log odds ratio for a 2x2 table.
        """
    f = self.table.flatten()
    return np.dot(np.log(f), np.r_[1, -1, -1, 1])