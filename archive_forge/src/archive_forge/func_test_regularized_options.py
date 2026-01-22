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
def test_regularized_options():
    n = 100
    p = 5
    np.random.seed(3132)
    xmat = np.random.normal(size=(n, p))
    yvec = xmat.sum(1) + np.random.normal(size=n)
    model1 = OLS(yvec - 1, xmat)
    result1 = model1.fit_regularized(alpha=1.0, L1_wt=0.5)
    model2 = OLS(yvec, xmat, offset=1)
    result2 = model2.fit_regularized(alpha=1.0, L1_wt=0.5, start_params=np.zeros(5))
    assert_allclose(result1.params, result2.params)