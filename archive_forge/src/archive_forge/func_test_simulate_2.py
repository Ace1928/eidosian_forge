import os
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, structural, varmax
from statsmodels.tsa.statespace.simulation_smoother import (
def test_simulate_2(self):
    n = 10
    Z = self.model['design']
    T = self.model['transition']
    measurement_shocks = np.zeros((n, self.model.k_endog))
    state_shocks = np.ones((n, self.model.ssm.k_posdef))
    initial_state = np.ones(self.model.k_states) * 2.5
    obs, states = self.model.ssm.simulate(nsimulations=n, measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state)
    desired_obs = np.zeros((n, self.model.k_endog))
    desired_state = np.zeros((n, self.model.k_states))
    desired_state[0] = initial_state
    desired_obs[0] = np.dot(Z, initial_state)
    for i in range(1, n):
        desired_state[i] = np.dot(T, desired_state[i - 1]) + state_shocks[i]
        desired_obs[i] = np.dot(Z, desired_state[i])
    assert_allclose(obs, desired_obs)
    assert_allclose(states, desired_state)