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
@pytest.mark.parametrize('loss', LOSS_INSTANCES, ids=loss_instance_name)
@pytest.mark.parametrize('sample_weight', [None, 'range'])
def test_loss_intercept_only(loss, sample_weight):
    """Test that fit_intercept_only returns the argmin of the loss.

    Also test that the gradient is zero at the minimum.
    """
    n_samples = 50
    if not loss.is_multiclass:
        y_true = loss.link.inverse(np.linspace(-4, 4, num=n_samples))
    else:
        y_true = np.arange(n_samples).astype(np.float64) % loss.n_classes
        y_true[::5] = 0
    if sample_weight == 'range':
        sample_weight = np.linspace(0.1, 2, num=n_samples)
    a = loss.fit_intercept_only(y_true=y_true, sample_weight=sample_weight)

    def fun(x):
        if not loss.is_multiclass:
            raw_prediction = np.full(shape=n_samples, fill_value=x)
        else:
            raw_prediction = np.ascontiguousarray(np.broadcast_to(x, shape=(n_samples, loss.n_classes)))
        return loss(y_true=y_true, raw_prediction=raw_prediction, sample_weight=sample_weight)
    if not loss.is_multiclass:
        opt = minimize_scalar(fun, tol=1e-07, options={'maxiter': 100})
        grad = loss.gradient(y_true=y_true, raw_prediction=np.full_like(y_true, a), sample_weight=sample_weight)
        assert a.shape == tuple()
        assert a.dtype == y_true.dtype
        assert_all_finite(a)
        a == approx(opt.x, rel=1e-07)
        grad.sum() == approx(0, abs=1e-12)
    else:
        opt = minimize(fun, np.zeros(loss.n_classes), tol=1e-13, options={'maxiter': 100}, method='SLSQP', constraints=LinearConstraint(np.ones((1, loss.n_classes)), 0, 0))
        grad = loss.gradient(y_true=y_true, raw_prediction=np.tile(a, (n_samples, 1)), sample_weight=sample_weight)
        assert a.dtype == y_true.dtype
        assert_all_finite(a)
        assert_allclose(a, opt.x, rtol=5e-06, atol=1e-12)
        assert_allclose(grad.sum(axis=0), 0, atol=1e-12)