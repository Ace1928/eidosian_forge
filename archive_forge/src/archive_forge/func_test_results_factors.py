from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests import test_dynamic_factor_mq_monte_carlo
def test_results_factors(reset_randomstate):
    endog, _, _, _, _, _ = gen_dfm_data(k_endog=2, nobs=1000)
    mod_dfm = dynamic_factor_mq.DynamicFactorMQ(endog, factors=['global'], factor_multiplicities=2, standardize=False, idiosyncratic_ar1=False)
    res_dfm = mod_dfm.smooth(mod_dfm.start_params)
    assert_allclose(res_dfm.factors.smoothed, res_dfm.states.smoothed[['global.1', 'global.2']])
    assert_allclose(res_dfm.factors.smoothed_cov.values, res_dfm.states.smoothed_cov.values, atol=1e-12)