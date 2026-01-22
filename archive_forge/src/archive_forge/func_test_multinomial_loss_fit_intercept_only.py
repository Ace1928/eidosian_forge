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
def test_multinomial_loss_fit_intercept_only():
    """Test that fit_intercept_only returns the mean functional for CCE."""
    rng = np.random.RandomState(0)
    n_classes = 4
    loss = HalfMultinomialLoss(n_classes=n_classes)
    y_train = rng.randint(0, n_classes + 1, size=100).astype(np.float64)
    baseline_prediction = loss.fit_intercept_only(y_true=y_train)
    assert baseline_prediction.shape == (n_classes,)
    p = np.zeros(n_classes, dtype=y_train.dtype)
    for k in range(n_classes):
        p[k] = (y_train == k).mean()
    assert_allclose(baseline_prediction, np.log(p) - np.mean(np.log(p)))
    assert_allclose(baseline_prediction[None, :], loss.link.link(p[None, :]))
    for y_train in (np.zeros(shape=10), np.ones(shape=10)):
        y_train = y_train.astype(np.float64)
        baseline_prediction = loss.fit_intercept_only(y_true=y_train)
        assert baseline_prediction.dtype == y_train.dtype
        assert_all_finite(baseline_prediction)