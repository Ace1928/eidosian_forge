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
def test_default_time(self):
    endog, exog, group = load_data('gee_logistic_1.csv')
    T = np.zeros(len(endog))
    idx = set(group)
    for ii in idx:
        jj = np.flatnonzero(group == ii)
        T[jj] = lrange(len(jj))
    family = families.Binomial()
    va = cov_struct.Autoregressive(grid=False)
    md1 = gee.GEE(endog, exog, group, family=family, cov_struct=va)
    mdf1 = md1.fit()
    md2 = gee.GEE(endog, exog, group, time=T, family=family, cov_struct=va)
    mdf2 = md2.fit()
    assert_almost_equal(mdf1.params, mdf2.params, decimal=6)
    assert_almost_equal(mdf1.standard_errors(), mdf2.standard_errors(), decimal=6)