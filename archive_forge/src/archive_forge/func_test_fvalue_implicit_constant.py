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
def test_fvalue_implicit_constant():
    nobs = 100
    np.random.seed(2)
    x = np.random.randn(nobs, 1)
    x = ((x > 0) == [True, False]).astype(int)
    y = x.sum(1) + np.random.randn(nobs)
    from statsmodels.regression.linear_model import OLS, WLS
    res = OLS(y, x).fit(cov_type='HC1')
    assert_(np.isnan(res.fvalue))
    assert_(np.isnan(res.f_pvalue))
    res.summary()
    res = WLS(y, x).fit(cov_type='HC1')
    assert_(np.isnan(res.fvalue))
    assert_(np.isnan(res.f_pvalue))
    res.summary()