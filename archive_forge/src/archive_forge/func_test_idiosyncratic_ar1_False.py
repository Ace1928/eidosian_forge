from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests import test_dynamic_factor_mq_monte_carlo
def test_idiosyncratic_ar1_False(reset_randomstate):
    endog, loadings, phi, sigma2, idio_ar1, idio_var = gen_dfm_data(k_endog=10, nobs=1000)
    mod_base = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=1)
    mod_dfm = dynamic_factor_mq.DynamicFactorMQ(endog, factor_orders=1, standardize=False, idiosyncratic_ar1=False)
    mod_dfm_ar1 = dynamic_factor_mq.DynamicFactorMQ(endog, factor_orders=1, standardize=False, idiosyncratic_ar1=True)
    params = np.r_[loadings, idio_var, phi]
    params_dfm = np.r_[loadings, phi, sigma2, idio_var]
    params_dfm_ar1 = np.r_[loadings, phi, sigma2, idio_ar1, idio_var]
    llf_base = mod_base.loglike(params)
    llf_dfm = mod_dfm.loglike(params_dfm)
    llf_dfm_ar1 = mod_dfm_ar1.loglike(params_dfm_ar1)
    assert_allclose(llf_dfm_ar1, llf_dfm)
    assert_allclose(llf_dfm, llf_base)
    assert_allclose(llf_dfm_ar1, llf_base)
    res0_dfm = mod_dfm.smooth(params_dfm)
    res0_dfm_ar1 = mod_dfm_ar1.smooth(params_dfm_ar1)
    assert_allclose(res0_dfm.smoothed_measurement_disturbance, res0_dfm_ar1.smoothed_state[1:])
    assert_allclose(res0_dfm.smoothed_measurement_disturbance_cov, res0_dfm_ar1.smoothed_state_cov[1:, 1:, :])
    if not SKIP_MONTE_CARLO_TESTS:
        res_dfm = mod_dfm.fit()
        actual_dfm = res_dfm.params.copy()
        scalar = actual_dfm[0] / params_dfm[0]
        actual_dfm[11] *= scalar
        actual_dfm[:10] /= scalar
        assert_allclose(actual_dfm, params_dfm, atol=0.1)
        res_dfm_ar1 = mod_dfm_ar1.fit()
        actual_dfm_ar1 = res_dfm_ar1.params.copy()
        scalar = actual_dfm_ar1[0] / params_dfm[0]
        actual_dfm_ar1[11] *= scalar
        actual_dfm_ar1[:10] /= scalar
        assert_allclose(actual_dfm_ar1, params_dfm_ar1, atol=0.1)
        desired = np.r_[actual_dfm_ar1[:12], actual_dfm_ar1[-10:]]
        assert_allclose(actual_dfm, desired, atol=0.1)