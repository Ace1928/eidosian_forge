from statsmodels.compat.python import lrange
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import t as student_t
from statsmodels.datasets import longley
from statsmodels.regression.linear_model import (
from statsmodels.tools.tools import add_constant
def test_rsquared_adj_overfit(self):
    with warnings.catch_warnings(record=True):
        x = np.random.randn(5)
        y = np.random.randn(5, 6)
        results = OLS(x, y).fit()
        rsquared_adj = results.rsquared_adj
        assert_equal(rsquared_adj, np.nan)