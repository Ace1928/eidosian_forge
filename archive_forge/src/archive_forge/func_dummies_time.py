from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
import statsmodels.tools.data as data_util
from pandas import Index, MultiIndex
def dummies_time(self):
    self.dummy_sparse(level=1)
    return self._dummies