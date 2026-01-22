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
def test_regularized_poisson():
    np.random.seed(8735)
    ng, gs, p = (1000, 5, 5)
    x = np.random.normal(size=(ng * gs, p))
    r = 0.5
    x[:, 2] = r * x[:, 1] + np.sqrt(1 - r ** 2) * x[:, 2]
    lpr = 0.7 * (x[:, 1] - x[:, 3])
    mean = np.exp(lpr)
    y = np.random.poisson(mean)
    groups = np.kron(np.arange(ng), np.ones(gs))
    model = gee.GEE(y, x, groups=groups, family=families.Poisson())
    result = model.fit_regularized(1e-07)
    assert_allclose(result.params, 0.7 * np.r_[0, 1, 0, -1, 0], rtol=0.01, atol=0.12)