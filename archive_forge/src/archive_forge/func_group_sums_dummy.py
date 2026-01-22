from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
import statsmodels.tools.data as data_util
from pandas import Index, MultiIndex
def group_sums_dummy(x, group_dummy):
    """sum by groups given group dummy variable

    group_dummy can be either ndarray or sparse matrix
    """
    if data_util._is_using_ndarray_type(group_dummy, None):
        return np.dot(x.T, group_dummy)
    else:
        return x.T * group_dummy