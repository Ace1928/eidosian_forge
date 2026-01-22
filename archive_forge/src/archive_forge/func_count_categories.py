from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
import statsmodels.tools.data as data_util
from pandas import Index, MultiIndex
def count_categories(self, level=0):
    """
        Sets the attribute counts to equal the bincount of the (integer-valued)
        labels.
        """
    self.counts = np.bincount(self.labels[level])