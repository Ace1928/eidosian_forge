import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_allclose
from statsmodels import datasets
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
@pytest.mark.parametrize('use_exog', [False, True])
@pytest.mark.parametrize('trend', ['n', 'c', 't'])
@pytest.mark.parametrize('concentrate_scale', [False, True])
@pytest.mark.parametrize('measurement_error', [False, True])
def test_smoothed_state_obs_weights_sarimax(use_exog, trend, concentrate_scale, measurement_error):
    endog = np.array([[0.2, np.nan, 1.2, -0.3, -1.5]]).T
    exog = np.array([2, 5.3, -1, 3.4, 0.0]) if use_exog else None
    trend_params = [0.1]
    ar_params = [0.5]
    exog_params = [1.4]
    meas_err_params = [1.2]
    cov_params = [0.8]
    params = []
    if trend in ['c', 't']:
        params += trend_params
    if use_exog:
        params += exog_params
    params += ar_params
    if measurement_error:
        params += meas_err_params
    if not concentrate_scale:
        params += cov_params
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0), trend=trend, exog=exog if use_exog else None, concentrate_scale=concentrate_scale, measurement_error=measurement_error)
    prior_mean = np.array([-0.4])
    prior_cov = np.eye(1) * 1.2
    mod.ssm.initialize_known(prior_mean, prior_cov)
    res = mod.smooth(params)
    n = mod.nobs
    m = mod.k_states
    p = mod.k_endog
    desired = np.zeros((n, n, m, p)) * np.nan
    for j in range(n):
        for i in range(p):
            if np.isnan(endog[j, i]):
                desired[:, j, :, i] = np.nan
            else:
                y = endog.copy()
                y[j, i] += 1.0
                tmp_mod = sarimax.SARIMAX(y, order=(1, 0, 0), trend=trend, exog=exog if use_exog else None, concentrate_scale=concentrate_scale, measurement_error=measurement_error)
                tmp_mod.ssm.initialize_known(prior_mean, prior_cov)
                tmp_res = tmp_mod.smooth(params)
                desired[:, j, :, i] = tmp_res.smoothed_state.T - res.smoothed_state.T
    desired_state_intercept_weights = np.zeros((n, n, m, m)) * np.nan
    for j in range(n):
        for ell in range(m):
            tmp_mod = sarimax.SARIMAX(endog, order=(1, 0, 0), trend=trend, exog=exog if use_exog else None, concentrate_scale=concentrate_scale, measurement_error=measurement_error)
            tmp_mod.ssm.initialize_known(prior_mean, prior_cov)
            tmp_mod.update(params)
            if tmp_mod['state_intercept'].ndim == 1:
                si = tmp_mod['state_intercept']
                tmp_mod['state_intercept'] = np.zeros((mod.k_states, mod.nobs))
                tmp_mod['state_intercept', :, :] = si
            tmp_mod['state_intercept', ell, j] += 1.0
            tmp_res = tmp_mod.ssm.smooth()
            desired_state_intercept_weights[:, j, :, ell] = tmp_res.smoothed_state.T - res.smoothed_state.T
    desired_prior_weights = np.zeros((n, m, m)) * np.nan
    for i in range(m):
        a = prior_mean.copy()
        a[i] += 1
        tmp_mod = sarimax.SARIMAX(endog, order=(1, 0, 0), trend=trend, exog=exog if use_exog else None, concentrate_scale=concentrate_scale, measurement_error=measurement_error)
        tmp_mod.ssm.initialize_known(a, prior_cov)
        tmp_res = tmp_mod.smooth(params)
        desired_prior_weights[:, :, i] = tmp_res.smoothed_state.T - res.smoothed_state.T
    mod.ssm.initialize_known(prior_mean, prior_cov)
    actual, actual_state_intercept_weights, actual_prior_weights = tools.compute_smoothed_state_weights(res)
    assert_allclose(actual, desired, atol=1e-08)
    assert_allclose(actual_state_intercept_weights, desired_state_intercept_weights, atol=1e-12)
    assert_allclose(actual_prior_weights, desired_prior_weights, atol=1e-12)