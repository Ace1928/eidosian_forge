from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
import os
from statsmodels import datasets
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from numpy.testing import assert_equal, assert_allclose
from . import kfas_helpers
def test_common_level_analytic():
    mod = model_common_level()
    y11, y21 = mod.endog[:, 0]
    theta = mod['design', 1, 0]
    sigma2_mu = mod['state_cov', 0, 0]
    res = mod.smooth()
    assert_allclose(res.predicted_state_cov[..., 0], np.zeros((2, 2)))
    assert_allclose(res.predicted_diffuse_state_cov[..., 0], np.eye(2))
    assert_allclose(res.predicted_state[:, 1], [y11, y21 - theta * y11])
    P2 = np.array([[1 + sigma2_mu, -theta], [-theta, 1 + theta ** 2]])
    assert_allclose(res.predicted_state_cov[..., 1], P2)
    assert_allclose(res.predicted_diffuse_state_cov[..., 1], np.zeros((2, 2)))
    assert_equal(res.nobs_diffuse, 1)