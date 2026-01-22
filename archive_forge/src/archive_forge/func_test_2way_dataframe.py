import numpy as np
from numpy.testing import (
import pytest
from scipy import stats
from statsmodels.datasets import macrodata
from statsmodels.regression.linear_model import OLS, WLS
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.sm_exceptions import InvalidTestWarning
from statsmodels.tools.tools import add_constant
from .results import (
def test_2way_dataframe(self):
    import pandas as pd
    long_groups = self.groups.reshape(-1, 1)
    groups2 = pd.DataFrame(np.hstack((long_groups, long_groups)))
    res = self.res1.get_robustcov_results('cluster', groups=groups2, use_correction=True, use_t=True)