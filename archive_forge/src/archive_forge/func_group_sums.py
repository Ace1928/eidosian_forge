from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
import statsmodels.tools.data as data_util
from pandas import Index, MultiIndex
def group_sums(self, x, use_bincount=True):
    return group_sums(x, self.group_int, use_bincount=use_bincount)