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
@pytest.mark.smoke
def test_t_test_summary_frame(self):
    res1 = self.res1
    mat = np.eye(len(res1.params))
    tt = res1.t_test(mat, cov_p=self.cov_robust)
    tt.summary_frame()