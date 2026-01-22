from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests import test_dynamic_factor_mq_monte_carlo
def check_extend(res1, res2, endog_M2, endog_Q2):
    res1_extend = res1.extend(endog_M2, endog_quarterly=endog_Q2)
    mod1_extend = res1_extend.model
    mod1 = res1.model
    check_identical_models(mod1, mod1_extend, check_nobs=False)
    assert_equal(mod1_extend.nobs, len(endog_M2))
    res2_extend = res2.extend(endog_M2, endog_quarterly=endog_Q2)
    mod2_extend = res2_extend.model
    mod2 = res2.model
    mod2_extend.update(res2_extend.params)
    mod2_extend['obs_intercept'] = mod2['obs_intercept']
    mod2_extend['design'] = mod2['design']
    mod2_extend['obs_cov'] = mod2['obs_cov']
    mod2_extend.update = lambda params, **kwargs: params
    res2_extend = mod2_extend.smooth(res2_extend.params)
    check_identical_models(mod2, mod2_extend, check_nobs=False)
    check_standardized_results(res1_extend, res2_extend, check_diagnostics=False)