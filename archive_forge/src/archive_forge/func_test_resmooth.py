import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_allclose
from statsmodels import datasets
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
def test_resmooth():
    endog = [0.1, -0.3, -0.1, 0.5, 0.02]
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0), measurement_error=True)
    res1 = mod.smooth([0.5, 2.0, 1.0])
    weights1_original, _, _ = tools.compute_smoothed_state_weights(res1, resmooth=False)
    res2 = mod.smooth([0.2, 1.0, 1.2])
    weights2_original, _, _ = tools.compute_smoothed_state_weights(res2, resmooth=False)
    weights1_no_resmooth, _, _ = tools.compute_smoothed_state_weights(res1, resmooth=False)
    weights1_resmooth, _, _ = tools.compute_smoothed_state_weights(res1, resmooth=True)
    weights2_no_resmooth, _, _ = tools.compute_smoothed_state_weights(res2, resmooth=False)
    weights2_resmooth, _, _ = tools.compute_smoothed_state_weights(res2, resmooth=True)
    weights1_default, _, _ = tools.compute_smoothed_state_weights(res1)
    weights2_default, _, _ = tools.compute_smoothed_state_weights(res2)
    assert_allclose(weights1_no_resmooth, weights2_original)
    assert_allclose(weights1_resmooth, weights1_original)
    assert_allclose(weights1_default, weights1_original)
    assert_allclose(weights2_no_resmooth, weights1_original)
    assert_allclose(weights2_resmooth, weights2_original)
    assert_allclose(weights2_default, weights2_original)