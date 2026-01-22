from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
import statsmodels.tools.data as data_util
from pandas import Index, MultiIndex
def dummies_groups(self, level=0):
    self.dummy_sparse(level=level)
    return self._dummies