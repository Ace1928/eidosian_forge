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
def test_margins_poisson(self):
    np.random.seed(34234)
    endog = np.r_[10, 15, 12, 13, 20, 18, 26, 29]
    exog = np.ones((8, 2))
    exog[:, 1] = np.r_[0, 0, 0, 0, 1, 1, 1, 1]
    groups = np.arange(8)
    model = gee.GEE(endog, exog, groups, family=families.Poisson())
    result = model.fit(cov_type='naive', start_params=[2.52572864, 0.62057649])
    marg = result.get_margeff()
    assert_allclose(marg.margeff, np.r_[11.0928], rtol=1e-06)
    assert_allclose(marg.margeff_se, np.r_[3.269015], rtol=1e-06)