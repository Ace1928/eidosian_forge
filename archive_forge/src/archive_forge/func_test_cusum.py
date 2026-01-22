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
def test_cusum():
    mod = RecursiveLS(endog, exog)
    res = mod.fit()
    d = res.nobs_diffuse
    cusum = res.cusum * np.std(res.resid_recursive[d:], ddof=1)
    cusum -= res.resid_recursive[d]
    cusum /= np.std(res.resid_recursive[d + 1:], ddof=1)
    cusum = cusum[1:]
    assert_allclose(cusum, results_stata.iloc[3:]['cusum'], atol=1e-06, rtol=1e-05)
    mod_ols = OLS(endog, exog)
    res_ols = mod_ols.fit()
    desired_cusum = recursive_olsresiduals(res_ols)[-2][1:]
    assert_allclose(res.cusum, desired_cusum, rtol=1e-06)
    actual_bounds = res._cusum_significance_bounds(alpha=0.05, ddof=1, points=np.arange(d + 1, res.nobs))
    desired_bounds = results_stata.iloc[3:][['lw', 'uw']].T
    assert_allclose(actual_bounds, desired_bounds, rtol=1e-06)
    actual_bounds = res._cusum_significance_bounds(alpha=0.05, ddof=0, points=np.arange(d, res.nobs))
    desired_bounds = recursive_olsresiduals(res_ols)[-1]
    assert_allclose(actual_bounds, desired_bounds)
    assert_raises(ValueError, res._cusum_squares_significance_bounds, alpha=0.123)