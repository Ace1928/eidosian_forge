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
@pytest.mark.parametrize('reg', [False, True])
def test_quasipoisson(reg):
    np.random.seed(343)
    n = 1000
    x = np.random.normal(size=(n, 3))
    g = np.random.gamma(1, 1, size=n)
    y = np.random.poisson(g)
    grp = np.kron(np.arange(100), np.ones(n // 100))
    model1 = gee.GEE(y, x, family=families.Poisson(), groups=grp)
    model2 = gee.GEE(y, x, family=families.Poisson(), groups=grp)
    if reg:
        result1 = model1.fit_regularized(pen_wt=0.1)
        result2 = model2.fit_regularized(pen_wt=0.1, scale='X2')
    else:
        result1 = model1.fit(cov_type='naive')
        result2 = model2.fit(scale='X2', cov_type='naive')
    assert_allclose(result1.params, result2.params)
    if not reg:
        assert_allclose(result2.cov_naive / result1.cov_naive, result2.scale * np.ones_like(result2.cov_naive))