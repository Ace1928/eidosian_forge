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
def test_compare_score_test_warnings(self):
    np.random.seed(6432)
    n = 200
    exog = np.random.normal(size=(n, 4))
    group = np.kron(np.arange(n / 4), np.ones(4))
    exog_sub = exog[:, [0, 3]]
    endog = exog_sub.sum(1) + 3 * np.random.normal(size=n)
    with assert_warns(UserWarning):
        mod_sub = gee.GEE(endog, exog_sub, group, cov_struct=cov_struct.Exchangeable())
        res_sub = mod_sub.fit()
        mod = gee.GEE(endog, exog, group, cov_struct=cov_struct.Independence())
        mod.compare_score_test(res_sub)
    with assert_warns(UserWarning):
        mod_sub = gee.GEE(endog, exog_sub, group, family=families.Gaussian())
        res_sub = mod_sub.fit()
        mod = gee.GEE(endog, exog, group, family=families.Poisson())
        mod.compare_score_test(res_sub)
    with assert_raises(Exception):
        mod_sub = gee.GEE(endog, exog_sub, group)
        res_sub = mod_sub.fit()
        mod = gee.GEE(endog[0:100], exog[:100, :], group[0:100])
        mod.compare_score_test(res_sub)
    with assert_warns(UserWarning):
        w = np.random.uniform(size=n)
        mod_sub = gee.GEE(endog, exog_sub, group, weights=w)
        res_sub = mod_sub.fit()
        mod = gee.GEE(endog, exog, group)
        mod.compare_score_test(res_sub)
    with pytest.warns(UserWarning):
        w = np.random.uniform(size=n)
        mod_sub = gee.GEE(endog, exog, group)
        res_sub = mod_sub.fit()
        mod = gee.GEE(endog, exog, group)
        mod.compare_score_test(res_sub)