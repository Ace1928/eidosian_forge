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
def test_score_test_ols():
    from statsmodels.regression.linear_model import OLS
    np.random.seed(5)
    nobs = 100
    sige = 0.5
    x = np.random.uniform(0, 1, size=(nobs, 5))
    x[:, 0] = 1
    beta = 1.0 / np.arange(1.0, x.shape[1] + 1)
    y = x.dot(beta) + sige * np.random.randn(nobs)
    res_ols = OLS(y, x).fit()
    res_olsc = OLS(y, x[:, :-2]).fit()
    co = res_ols.compare_lm_test(res_olsc, demean=False)
    res_glm = GLM(y, x[:, :-2], family=sm.families.Gaussian()).fit()
    co2 = res_glm.model.score_test(res_glm.params, exog_extra=x[:, -2:])
    assert_allclose(co[0] * 97 / 100.0, co2[0], rtol=1e-13)