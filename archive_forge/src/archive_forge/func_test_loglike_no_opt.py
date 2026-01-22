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
def test_loglike_no_opt():
    y = np.asarray([0, 1, 0, 0, 1, 1, 0, 1, 1, 1])
    x = np.arange(10, dtype=np.float64)

    def llf(params):
        lin_pred = params[0] + params[1] * x
        pr = 1 / (1 + np.exp(-lin_pred))
        return np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))
    for params in ([0, 0], [0, 1], [0.5, 0.5]):
        mod = sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial())
        res = mod.fit(start_params=params, maxiter=0)
        like = llf(params)
        assert_almost_equal(like, res.llf)