import os
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.tests.results import results_kalman_filter
from statsmodels.tsa.statespace.sarimax import SARIMAX
def test_time_varying_transition():
    endog = np.array([10, 5, 2.5, 1.25, 2.5, 5, 10])
    transition = np.ones((1, 1, 7))
    transition[..., :5] = 0.5
    transition[..., 5:] = 2
    mod1 = SARIMAX(endog, order=(1, 0, 0), measurement_error=True)
    mod1.update([2.0, 1.0, 1.0])
    mod1.ssm['transition'] = transition
    res1 = mod1.ssm.smooth()
    mod2 = SARIMAX(endog, order=(1, 0, 0), measurement_error=True)
    mod2.ssm.filter_univariate = True
    mod2.update([2.0, 1.0, 1.0])
    mod2.ssm['transition'] = transition
    res2 = mod2.ssm.smooth()
    n_disturbance_variates = (mod1.k_endog + mod1.k_posdef) * mod1.nobs
    sim1 = mod1.simulation_smoother(disturbance_variates=np.zeros(n_disturbance_variates), initial_state_variates=np.zeros(mod1.k_states))
    sim2 = mod2.simulation_smoother(disturbance_variates=np.zeros(n_disturbance_variates), initial_state_variates=np.zeros(mod2.k_states))
    assert_allclose(res1.forecasts[0, :], res2.forecasts[0, :])
    assert_allclose(res1.forecasts_error[0, :], res2.forecasts_error[0, :])
    assert_allclose(res1.forecasts_error_cov[0, 0, :], res2.forecasts_error_cov[0, 0, :])
    assert_allclose(res1.filtered_state, res2.filtered_state)
    assert_allclose(res1.filtered_state_cov, res2.filtered_state_cov)
    assert_allclose(res1.predicted_state, res2.predicted_state)
    assert_allclose(res1.predicted_state_cov, res2.predicted_state_cov)
    assert_allclose(res1.llf_obs, res2.llf_obs)
    assert_allclose(res1.smoothed_state, res2.smoothed_state)
    assert_allclose(res1.smoothed_state_cov, res2.smoothed_state_cov)
    assert_allclose(res1.smoothed_measurement_disturbance, res2.smoothed_measurement_disturbance)
    assert_allclose(res1.smoothed_measurement_disturbance_cov.diagonal(), res2.smoothed_measurement_disturbance_cov.diagonal())
    assert_allclose(res1.smoothed_state_disturbance, res2.smoothed_state_disturbance)
    assert_allclose(res1.smoothed_state_disturbance_cov, res2.smoothed_state_disturbance_cov)
    assert_allclose(sim1.simulated_state, sim2.simulated_state)
    assert_allclose(sim1.simulated_measurement_disturbance, sim2.simulated_measurement_disturbance)
    assert_allclose(sim1.simulated_state_disturbance, sim2.simulated_state_disturbance)