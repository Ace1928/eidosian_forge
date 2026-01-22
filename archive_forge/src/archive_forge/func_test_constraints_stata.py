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
def test_constraints_stata():
    endog = dta['infl']
    exog = add_constant(dta[['m1', 'unemp']])
    mod = RecursiveLS(endog, exog, constraints='m1 + unemp = 1')
    res = mod.fit()
    desired = [-0.7001083844336, -0.001847751406, 1.001847751406]
    assert_allclose(res.params, desired)
    desired = [0.4699552366, 0.0005369357, 0.0005369357]
    bse = np.asarray(res.bse)
    assert_allclose(bse[0], desired[0], atol=0.1)
    assert_allclose(bse[1:], desired[1:], atol=0.0001)
    desired = -534.4292052931121
    scale_alternative = np.sum((res.standardized_forecasts_error[0, 1:] * res.filter_results.obs_cov[0, 0] ** 0.5) ** 2) / mod.nobs
    llf_alternative = np.log(norm.pdf(res.resid_recursive, loc=0, scale=scale_alternative ** 0.5)).sum()
    assert_allclose(llf_alternative, desired)