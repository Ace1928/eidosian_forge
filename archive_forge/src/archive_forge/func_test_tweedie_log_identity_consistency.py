import pickle
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx
from scipy.optimize import (
from scipy.special import logsumexp
from sklearn._loss.link import IdentityLink, _inclusive_low_high
from sklearn._loss.loss import (
from sklearn.utils import _IS_WASM, assert_all_finite
from sklearn.utils._testing import create_memmap_backed_data, skip_if_32bit
@pytest.mark.parametrize('p', [-1.5, 0, 1, 1.5, 2, 3])
def test_tweedie_log_identity_consistency(p):
    """Test for identical losses when only the link function is different."""
    half_tweedie_log = HalfTweedieLoss(power=p)
    half_tweedie_identity = HalfTweedieLossIdentity(power=p)
    n_samples = 10
    y_true, raw_prediction = random_y_true_raw_prediction(loss=half_tweedie_log, n_samples=n_samples, seed=42)
    y_pred = half_tweedie_log.link.inverse(raw_prediction)
    loss_log = half_tweedie_log.loss(y_true=y_true, raw_prediction=raw_prediction) + half_tweedie_log.constant_to_optimal_zero(y_true)
    loss_identity = half_tweedie_identity.loss(y_true=y_true, raw_prediction=y_pred) + half_tweedie_identity.constant_to_optimal_zero(y_true)
    assert_allclose(loss_log, loss_identity)
    gradient_log, hessian_log = half_tweedie_log.gradient_hessian(y_true=y_true, raw_prediction=raw_prediction)
    gradient_identity, hessian_identity = half_tweedie_identity.gradient_hessian(y_true=y_true, raw_prediction=y_pred)
    assert_allclose(gradient_log, y_pred * gradient_identity)
    assert_allclose(hessian_log, y_pred * gradient_identity + y_pred ** 2 * hessian_identity)