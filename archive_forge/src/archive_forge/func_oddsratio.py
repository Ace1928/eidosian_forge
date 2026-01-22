import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def oddsratio(self):
    """
        Returns the odds ratio for a 2x2 table.
        """
    return self.table[0, 0] * self.table[1, 1] / (self.table[0, 1] * self.table[1, 0])