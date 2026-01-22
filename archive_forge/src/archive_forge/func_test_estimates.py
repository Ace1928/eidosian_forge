import os
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises
import pandas as pd
import pytest
from scipy.stats import norm
from statsmodels.datasets import macrodata
from statsmodels.genmod.api import GLM
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.recursive_ls import RecursiveLS
from statsmodels.stats.diagnostic import recursive_olsresiduals
from statsmodels.tools import add_constant
from statsmodels.tools.eval_measures import aic, bic
from statsmodels.tools.sm_exceptions import ValueWarning
def test_estimates():
    mod = RecursiveLS(endog, exog)
    res = mod.fit()
    assert_equal(mod.start_params, 0)
    assert_allclose(res.recursive_coefficients.filtered[:, 2:10].T, results_R.iloc[:8][['beta1', 'beta2']], rtol=1e-05)
    assert_allclose(res.recursive_coefficients.filtered[:, 9:20].T, results_R.iloc[7:18][['beta1', 'beta2']])
    assert_allclose(res.recursive_coefficients.filtered[:, 19:].T, results_R.iloc[17:][['beta1', 'beta2']])
    mod_ols = OLS(endog, exog)
    res_ols = mod_ols.fit()
    assert_allclose(res.params, res_ols.params)