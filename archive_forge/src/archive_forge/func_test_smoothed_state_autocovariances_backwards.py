import os
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, varmax
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from statsmodels.tsa.statespace.kalman_filter import FILTER_UNIVARIATE
from statsmodels.tsa.statespace.kalman_smoother import (
@pytest.mark.parametrize('missing', ['all', 'partial', 'mixed', None])
@pytest.mark.parametrize('filter_univariate', [True, False])
@pytest.mark.parametrize('tvp', [True, False])
def test_smoothed_state_autocovariances_backwards(missing, filter_univariate, tvp):
    """
    Test for Cov(t, t - lag)
    """
    _, res = get_acov_model(missing, filter_univariate, tvp)
    cov = res.smoothed_state_cov.transpose(2, 0, 1)
    desired_acov1 = cov[:, :2, 2:4]
    desired_acov2 = cov[:, :2, 4:6]
    desired_acov3 = cov[:, :2, 6:8]
    acov1 = res.smoothed_state_autocovariance(1).transpose(2, 0, 1)
    assert_allclose(acov1[1:, :2, :2], desired_acov1[1:], rtol=1e-06, atol=1e-06)
    assert_equal(acov1[:1], np.nan)
    acov2 = res.smoothed_state_autocovariance(2).transpose(2, 0, 1)
    assert_allclose(acov2[2:, :2, :2], desired_acov2[2:], rtol=1e-06, atol=1e-06)
    assert_equal(acov2[:2], np.nan)
    acov3 = res.smoothed_state_autocovariance(3).transpose(2, 0, 1)
    assert_allclose(acov3[3:, :2, :2], desired_acov3[3:], rtol=1e-06, atol=1e-06)
    assert_equal(acov3[:3], np.nan)
    acov1 = res.smoothed_state_autocovariance(1, t=0)
    assert_allclose(acov1, np.nan)
    acov1 = res.smoothed_state_autocovariance(1, t=1)
    assert_allclose(acov1[:2, :2], desired_acov1[1], rtol=1e-06, atol=1e-06)
    acov1 = res.smoothed_state_autocovariance(1, start=8, end=9).transpose(2, 0, 1)
    assert_allclose(acov1[:, :2, :2], desired_acov1[8:9], rtol=1e-06, atol=1e-06)
    acov2 = res.smoothed_state_autocovariance(2, t=0)
    assert_allclose(acov2, np.nan)
    acov2 = res.smoothed_state_autocovariance(2, t=1)
    assert_allclose(acov2, np.nan)
    acov2 = res.smoothed_state_autocovariance(2, t=2)
    assert_allclose(acov2[:2, :2], desired_acov2[2], rtol=1e-06, atol=1e-06)
    acov2 = res.smoothed_state_autocovariance(2, start=8, end=9).transpose(2, 0, 1)
    assert_allclose(acov2[:, :2, :2], desired_acov2[8:9], rtol=1e-06, atol=1e-06)