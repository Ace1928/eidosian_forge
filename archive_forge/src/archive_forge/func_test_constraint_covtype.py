from statsmodels.compat import lrange
import os
import numpy as np
import pytest
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
import statsmodels.genmod.generalized_estimating_equations as gee
import statsmodels.tools as tools
import statsmodels.regression.linear_model as lm
from statsmodels.genmod import families
from statsmodels.genmod import cov_struct
import statsmodels.discrete.discrete_model as discrete
import pandas as pd
from scipy.stats.distributions import norm
import warnings
def test_constraint_covtype(self):
    np.random.seed(6432)
    n = 200
    exog = np.random.normal(size=(n, 4))
    endog = exog[:, 0] + exog[:, 1] + exog[:, 2]
    endog += 3 * np.random.normal(size=n)
    group = np.kron(np.arange(n / 4), np.ones(4))
    L = np.array([[1.0, -1, 0, 0]])
    R = np.array([0.0])
    family = families.Gaussian()
    va = cov_struct.Independence()
    for cov_type in ('robust', 'naive', 'bias_reduced'):
        model = gee.GEE(endog, exog, group, family=family, cov_struct=va, constraint=(L, R))
        result = model.fit(cov_type=cov_type)
        result.standard_errors(cov_type=cov_type)
        assert_allclose(result.cov_robust.shape, np.r_[4, 4])
        assert_allclose(result.cov_naive.shape, np.r_[4, 4])
        if cov_type == 'bias_reduced':
            assert_allclose(result.cov_robust_bc.shape, np.r_[4, 4])