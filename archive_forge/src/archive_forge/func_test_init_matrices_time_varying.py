import os
import warnings
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace.representation import Representation
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.simulation_smoother import SimulationSmoother
from statsmodels.tsa.statespace import tools, sarimax
from .results import results_kalman_filter
from numpy.testing import (
def test_init_matrices_time_varying():
    nobs = 10
    k_endog = 2
    k_states = 3
    k_posdef = 1
    endog = np.zeros((10, 2))
    obs_intercept = np.reshape(np.arange(k_endog * nobs) * 1.0, (k_endog, nobs))
    design = np.reshape(np.arange(k_endog * k_states * nobs) * 1.0, (k_endog, k_states, nobs))
    obs_cov = np.reshape(np.arange(k_endog ** 2 * nobs) * 1.0, (k_endog, k_endog, nobs))
    state_intercept = np.reshape(np.arange(k_states * nobs) * 1.0, (k_states, nobs))
    transition = np.reshape(np.arange(k_states ** 2 * nobs) * 1.0, (k_states, k_states, nobs))
    selection = np.reshape(np.arange(k_states * k_posdef * nobs) * 1.0, (k_states, k_posdef, nobs))
    state_cov = np.reshape(np.arange(k_posdef ** 2 * nobs) * 1.0, (k_posdef, k_posdef, nobs))
    mod = Representation(endog, k_states=k_states, k_posdef=k_posdef, obs_intercept=obs_intercept, design=design, obs_cov=obs_cov, state_intercept=state_intercept, transition=transition, selection=selection, state_cov=state_cov)
    assert_allclose(mod['obs_intercept'], obs_intercept)
    assert_allclose(mod['design'], design)
    assert_allclose(mod['obs_cov'], obs_cov)
    assert_allclose(mod['state_intercept'], state_intercept)
    assert_allclose(mod['transition'], transition)
    assert_allclose(mod['selection'], selection)
    assert_allclose(mod['state_cov'], state_cov)