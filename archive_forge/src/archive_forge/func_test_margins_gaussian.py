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
def test_margins_gaussian(self):
    n = 40
    np.random.seed(34234)
    exog = np.random.normal(size=(n, 3))
    exog[:, 0] = 1
    groups = np.kron(np.arange(n / 4), np.r_[1, 1, 1, 1])
    endog = exog[:, 1] + np.random.normal(size=n)
    model = gee.GEE(endog, exog, groups)
    result = model.fit(start_params=[-0.000488085602, 1.18501903, 0.04788201])
    marg = result.get_margeff()
    assert_allclose(marg.margeff, result.params[1:])
    assert_allclose(marg.margeff_se, result.bse[1:])
    marg.summary()