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
def test_wls_tss():
    y = np.array([22, 22, 22, 23, 23, 23])
    x = [[1, 0], [1, 0], [1, 1], [0, 1], [0, 1], [0, 1]]
    ols_mod = OLS(y, add_constant(x, prepend=False)).fit()
    yw = np.array([22, 22, 23.0])
    Xw = [[1, 0], [1, 1], [0, 1]]
    w = np.array([2, 1, 3.0])
    wls_mod = WLS(yw, add_constant(Xw, prepend=False), weights=w).fit()
    assert_equal(ols_mod.centered_tss, wls_mod.centered_tss)