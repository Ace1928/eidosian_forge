import os
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, structural, varmax
from statsmodels.tsa.statespace.simulation_smoother import (
def test_simulation_smoothing_state_intercept():
    nobs = 10
    intercept = 100
    endog = np.ones(nobs) * intercept
    mod = sarimax.SARIMAX(endog, order=(0, 0, 0), trend='c', measurement_error=True)
    mod.initialize_known([100], [[0]])
    mod.update([intercept, 1.0, 1.0])
    sim = mod.simulation_smoother()
    sim.simulate(measurement_disturbance_variates=np.zeros(mod.nobs), state_disturbance_variates=np.zeros(mod.nobs), initial_state_variates=np.zeros(1))
    assert_equal(sim.simulated_state[0], intercept)