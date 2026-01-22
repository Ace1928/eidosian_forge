import numpy as np
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.tsa.statespace.simulation_smoother import (
from numpy.testing import assert_equal
def test_smoother_outputs(self):
    model = self.model
    model.smoother_output = 0
    model.smoother_state = True
    assert_equal(model.smoother_output, SMOOTHER_STATE)
    model.smoother_disturbance = True
    assert_equal(model.smoother_output, SMOOTHER_STATE | SMOOTHER_DISTURBANCE)
    model.smoother_state = False
    assert_equal(model.smoother_output, SMOOTHER_DISTURBANCE)
    model.set_smoother_output(SMOOTHER_DISTURBANCE_COV)
    assert_equal(model.smoother_output, SMOOTHER_DISTURBANCE_COV)
    model.set_smoother_output(smoother_disturbance=True, smoother_disturbance_cov=False)
    assert_equal(model.smoother_output, SMOOTHER_DISTURBANCE)
    model.smoother_output = 0
    for name in model.smoother_outputs:
        if name == 'smoother_all':
            continue
        setattr(model, name, True)
    assert_equal(model.smoother_output, SMOOTHER_STATE | SMOOTHER_STATE_COV | SMOOTHER_STATE_AUTOCOV | SMOOTHER_DISTURBANCE | SMOOTHER_DISTURBANCE_COV)
    assert_equal(model.smoother_output, SMOOTHER_ALL)
    for name in model.smoother_outputs:
        if name == 'smoother_all':
            continue
        setattr(model, name, False)
    assert_equal(model.smoother_output, 0)