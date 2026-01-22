import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.datasets import cpunish, longley
from statsmodels.discrete import discrete_model as discrete
from statsmodels.genmod.generalized_linear_model import GLM, SET_USE_BIC_LLF
from statsmodels.tools.numdiff import (
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import add_constant
def test_poisson_deviance():
    np.random.seed(123987)
    nobs, k_vars = (50, 3 - 1)
    x = sm.add_constant(np.random.randn(nobs, k_vars))
    mu_true = np.exp(x.sum(1))
    y = np.random.poisson(mu_true, size=nobs)
    mod = sm.GLM(y, x[:, :], family=sm.genmod.families.Poisson())
    res = mod.fit()
    d_i = res.resid_deviance
    d = res.deviance
    lr = (mod.family.loglike(y, y + 1e-20) - mod.family.loglike(y, res.fittedvalues)) * 2
    assert_allclose(d, (d_i ** 2).sum(), rtol=1e-12)
    assert_allclose(d, lr, rtol=1e-12)
    mod_nc = sm.GLM(y, x[:, 1:], family=sm.genmod.families.Poisson())
    res_nc = mod_nc.fit()
    d_i = res_nc.resid_deviance
    d = res_nc.deviance
    lr = (mod.family.loglike(y, y + 1e-20) - mod.family.loglike(y, res_nc.fittedvalues)) * 2
    assert_allclose(d, (d_i ** 2).sum(), rtol=1e-12)
    assert_allclose(d, lr, rtol=1e-12)