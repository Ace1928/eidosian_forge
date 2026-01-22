import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.stats.distributions import norm
from statsmodels.base._penalties import Penalty
import statsmodels.base.model as base
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
def group_list(self, array):
    """
        Returns `array` split into subarrays corresponding to the
        grouping structure.
        """
    if array is None:
        return None
    if array.ndim == 1:
        return [np.array(array[self.row_indices[k]]) for k in self.group_labels]
    else:
        return [np.array(array[self.row_indices[k], :]) for k in self.group_labels]