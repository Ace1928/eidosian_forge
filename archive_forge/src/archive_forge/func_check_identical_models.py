from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests import test_dynamic_factor_mq_monte_carlo
def check_identical_models(mod1, mod2, check_nobs=True):
    if check_nobs:
        assert_equal(mod2.nobs, mod1.nobs)
    assert_equal(mod2.k_endog, mod1.k_endog)
    assert_equal(mod2.k_endog_M, mod1.k_endog_M)
    assert_equal(mod2.k_endog_Q, mod1.k_endog_Q)
    assert_equal(mod2.k_states, mod1.k_states)
    assert_equal(mod2.ssm.k_posdef, mod1.ssm.k_posdef)
    assert_allclose(mod2._endog_mean, mod1._endog_mean)
    assert_allclose(mod2._endog_std, mod1._endog_std)
    assert_allclose(mod2.standardize, mod1.standardize)
    assert_equal(mod2.factors, mod1.factors)
    assert_equal(mod2.factor_orders, mod1.factor_orders)
    assert_equal(mod2.factor_multiplicities, mod1.factor_multiplicities)
    assert_equal(mod2.idiosyncratic_ar1, mod1.idiosyncratic_ar1)
    assert_equal(mod2.init_t0, mod1.init_t0)
    assert_equal(mod2.obs_cov_diag, mod1.obs_cov_diag)
    assert_allclose(mod2.endog_factor_map, mod1.endog_factor_map)
    assert_allclose(mod2.factor_block_orders, mod1.factor_block_orders)
    assert_equal(mod2.endog_names, mod1.endog_names)
    assert_equal(mod2.factor_names, mod1.factor_names)
    assert_equal(mod2.k_factors, mod1.k_factors)
    assert_equal(mod2.k_factor_blocks, mod1.k_factor_blocks)
    assert_equal(mod2.max_factor_order, mod1.max_factor_order)