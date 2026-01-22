import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def cumulative_oddsratios(self):
    """
        Returns the cumulative odds ratios for a contingency table.

        See documentation for cumulative_log_oddsratio.
        """
    return np.exp(self.cumulative_log_oddsratios)