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
def test_compare_OLS(self):
    vs = cov_struct.Independence()
    family = families.Gaussian()
    np.random.seed(34234)
    Y = np.random.normal(size=100)
    X1 = np.random.normal(size=100)
    X2 = np.random.normal(size=100)
    X3 = np.random.normal(size=100)
    groups = np.kron(lrange(20), np.ones(5))
    D = pd.DataFrame({'Y': Y, 'X1': X1, 'X2': X2, 'X3': X3})
    md = gee.GEE.from_formula('Y ~ X1 + X2 + X3', groups, D, family=family, cov_struct=vs)
    mdf = md.fit()
    ols = lm.OLS.from_formula('Y ~ X1 + X2 + X3', data=D).fit()
    ols = ols._results
    assert_almost_equal(ols.params, mdf.params, decimal=10)
    se = mdf.standard_errors(cov_type='naive')
    assert_almost_equal(ols.bse, se, decimal=10)
    naive_tvalues = mdf.params / np.sqrt(np.diag(mdf.cov_naive))
    assert_almost_equal(naive_tvalues, ols.tvalues, decimal=10)