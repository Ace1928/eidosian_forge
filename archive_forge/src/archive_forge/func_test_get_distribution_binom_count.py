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
def test_get_distribution_binom_count(self):
    res1 = self.res1
    res_scale = 1
    mu_prob = res1.fittedvalues
    n = res1.model.n_trials
    distr = res1.model.family.get_distribution(mu_prob, res_scale, n_trials=n)
    var_endog = res1.model.family.variance(mu_prob) * res_scale
    m, v = distr.stats()
    assert_allclose(mu_prob * n, m, rtol=1e-13)
    assert_allclose(var_endog * n, v, rtol=1e-13)
    distr2 = res1.model.get_distribution(res1.params, res_scale, n_trials=n)
    for k in distr2.kwds:
        assert_allclose(distr.kwds[k], distr2.kwds[k], rtol=1e-13)