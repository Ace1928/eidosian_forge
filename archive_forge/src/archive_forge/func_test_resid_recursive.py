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
def test_resid_recursive():
    mod = RecursiveLS(endog, exog)
    res = mod.fit()
    assert_allclose(res.resid_recursive[2:10].T, results_R.iloc[:8]['rec_resid'])
    assert_allclose(res.resid_recursive[9:20].T, results_R.iloc[7:18]['rec_resid'])
    assert_allclose(res.resid_recursive[19:].T, results_R.iloc[17:]['rec_resid'])
    assert_allclose(res.resid_recursive[3:], results_stata.iloc[3:]['rr'], atol=1e-05, rtol=1e-05)
    mod_ols = OLS(endog, exog)
    res_ols = mod_ols.fit()
    desired_resid_recursive = recursive_olsresiduals(res_ols)[4][2:]
    assert_allclose(res.resid_recursive[2:], desired_resid_recursive)