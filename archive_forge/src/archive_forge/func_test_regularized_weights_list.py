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
def test_regularized_weights_list(self):
    np.random.seed(132)
    exog1 = np.random.normal(size=(100, 3))
    endog1 = exog1[:, 0] + exog1[:, 1] + np.random.normal(size=100)
    exog2 = np.random.normal(size=(100, 3))
    endog2 = exog2[:, 0] + exog2[:, 1] + np.random.normal(size=100)
    exog_a = np.vstack((exog1, exog1, exog2))
    endog_a = np.concatenate((endog1, endog1, endog2))
    exog_b = np.vstack((exog1, exog2))
    endog_b = np.concatenate((endog1, endog2))
    wgts = np.ones(200)
    wgts[0:100] = 2
    sigma = np.diag(1 / wgts)
    for L1_wt in (0, 0.5, 1):
        for alpha_element in (0, 1):
            alpha = [alpha_element] * 3
            mod1 = OLS(endog_a, exog_a)
            rslt1 = mod1.fit_regularized(L1_wt=L1_wt, alpha=alpha)
            mod2 = WLS(endog_b, exog_b, weights=wgts)
            rslt2 = mod2.fit_regularized(L1_wt=L1_wt, alpha=alpha)
            mod3 = GLS(endog_b, exog_b, sigma=sigma)
            rslt3 = mod3.fit_regularized(L1_wt=L1_wt, alpha=alpha)
            assert_almost_equal(rslt1.params, rslt2.params, decimal=3)
            assert_almost_equal(rslt1.params, rslt3.params, decimal=3)