import os
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, structural, varmax
from statsmodels.tsa.statespace.simulation_smoother import (
def test_simulate_0(self):
    n = 10
    measurement_shocks = np.zeros((n, self.model.k_endog))
    state_shocks = np.zeros((n, self.model.ssm.k_posdef))
    initial_state = np.zeros(self.model.k_states)
    obs, states = self.model.ssm.simulate(nsimulations=n, measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state)
    assert_allclose(obs, np.zeros((n, self.model.k_endog)))
    assert_allclose(states, np.zeros((n, self.model.k_states)))