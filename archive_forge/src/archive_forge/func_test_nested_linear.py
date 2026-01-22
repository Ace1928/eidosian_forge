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
def test_nested_linear(self):
    family = families.Gaussian()
    endog, exog, group = load_data('gee_nested_linear_1.csv')
    group_n = []
    for i in range(endog.shape[0] // 10):
        group_n.extend([0] * 5)
        group_n.extend([1] * 5)
    group_n = np.array(group_n)[:, None]
    dp = cov_struct.Independence()
    md = gee.GEE(endog, exog, group, None, family, dp)
    mdf1 = md.fit()
    cf = np.r_[-0.1671073, 1.00467426, -2.01723004, 0.97297106]
    se = np.r_[0.08629606, 0.04058653, 0.04067038, 0.03777989]
    assert_almost_equal(mdf1.params, cf, decimal=6)
    assert_almost_equal(mdf1.standard_errors(), se, decimal=6)
    ne = cov_struct.Nested()
    md = gee.GEE(endog, exog, group, None, family, ne, dep_data=group_n)
    mdf2 = md.fit(start_params=mdf1.params)
    cf = np.r_[-0.16655319, 1.02183688, -2.00858719, 1.00101969]
    se = np.r_[0.08632616, 0.02913582, 0.03114428, 0.02893991]
    assert_almost_equal(mdf2.params, cf, decimal=6)
    assert_almost_equal(mdf2.standard_errors(), se, decimal=6)
    smry = mdf2.cov_struct.summary()
    assert_allclose(smry.Variance, np.r_[1.043878, 0.611656, 1.421205], atol=1e-05, rtol=1e-05)