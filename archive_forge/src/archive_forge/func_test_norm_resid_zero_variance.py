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
def test_norm_resid_zero_variance(self):
    with warnings.catch_warnings(record=True):
        y = self.res1.model.endog
        res = OLS(y, y).fit()
        assert_allclose(res.scale, 0, atol=1e-20)
        assert_allclose(res.wresid, res.resid_pearson, atol=5e-11)