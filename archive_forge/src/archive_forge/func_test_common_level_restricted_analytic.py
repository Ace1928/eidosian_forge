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
def test_common_level_restricted_analytic():
    mod = model_common_level(restricted=True)
    y11, y21 = mod.endog[:, 0]
    theta = mod['design', 1, 0]
    sigma2_mu = mod['state_cov', 0, 0]
    res = mod.smooth()
    assert_allclose(res.predicted_state_cov[..., 0], 0)
    assert_allclose(res.predicted_diffuse_state_cov[..., 0], 1)
    phi = 1 / (1 + theta ** 2)
    assert_allclose(res.predicted_state[:, 1], phi * (y11 + theta * y21))
    assert_allclose(res.predicted_state_cov[..., 1], phi + sigma2_mu)
    assert_allclose(res.predicted_diffuse_state_cov[..., 1], 0)
    assert_equal(res.nobs_diffuse, 1)