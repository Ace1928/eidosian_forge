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
def test_regularized_refit():
    n = 100
    p = 5
    np.random.seed(3132)
    xmat = np.random.normal(size=(n, p))
    yvec = xmat[:, 0] + xmat[:, 2] + np.random.normal(size=n)
    model1 = OLS(yvec, xmat)
    result1 = model1.fit_regularized(alpha=2.0, L1_wt=0.5, refit=True)
    model2 = OLS(yvec, xmat[:, [0, 2]])
    result2 = model2.fit()
    ii = [0, 2]
    assert_allclose(result1.params[ii], result2.params)
    assert_allclose(result1.bse[ii], result2.bse)