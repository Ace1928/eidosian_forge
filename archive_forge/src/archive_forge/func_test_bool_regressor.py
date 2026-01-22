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
def test_bool_regressor(reset_randomstate):
    exog = np.random.randint(0, 2, size=(100, 2)).astype(bool)
    endog = np.random.standard_normal(100)
    bool_res = OLS(endog, exog).fit()
    res = OLS(endog, exog.astype(np.double)).fit()
    assert_allclose(bool_res.params, res.params)