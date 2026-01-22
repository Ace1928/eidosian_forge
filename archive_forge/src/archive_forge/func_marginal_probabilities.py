import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def marginal_probabilities(self):
    """
        Estimate marginal probability distributions for the rows and columns.

        Returns
        -------
        row : ndarray
            Marginal row probabilities
        col : ndarray
            Marginal column probabilities
        """
    n = self.table.sum()
    row = self.table.sum(1) / n
    col = self.table.sum(0) / n
    if isinstance(self.table_orig, pd.DataFrame):
        row = pd.Series(row, self.table_orig.index)
        col = pd.Series(col, self.table_orig.columns)
    return (row, col)