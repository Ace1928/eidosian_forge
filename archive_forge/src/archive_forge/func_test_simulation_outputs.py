import numpy as np
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.tsa.statespace.simulation_smoother import (
from numpy.testing import assert_equal
def test_simulation_outputs(self):
    assert_equal(self.model.get_simulation_output(SIMULATION_STATE), SIMULATION_STATE)
    assert_equal(self.model.get_simulation_output(simulate_state=True, simulate_disturbance=True), SIMULATION_ALL)