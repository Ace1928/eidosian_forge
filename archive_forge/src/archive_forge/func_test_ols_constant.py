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
def test_ols_constant(reset_randomstate):
    y = np.random.standard_normal(200)
    x = np.ones((200, 1))
    res = OLS(y, x).fit()
    with warnings.catch_warnings(record=True) as recording:
        assert np.isnan(res.fvalue)
        assert np.isnan(res.f_pvalue)
    assert len(recording) == 0