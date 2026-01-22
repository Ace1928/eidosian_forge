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
@pytest.mark.parametrize('loss', ALL_LOSSES)
@pytest.mark.parametrize('readonly_memmap', [False, True])
@pytest.mark.parametrize('dtype_in', [np.float32, np.float64])
@pytest.mark.parametrize('dtype_out', [np.float32, np.float64])
@pytest.mark.parametrize('sample_weight', [None, 1])
@pytest.mark.parametrize('out1', [None, 1])
@pytest.mark.parametrize('out2', [None, 1])
@pytest.mark.parametrize('n_threads', [1, 2])
def test_loss_dtype(loss, readonly_memmap, dtype_in, dtype_out, sample_weight, out1, out2, n_threads):
    """Test acceptance of dtypes, readonly and writeable arrays in loss functions.

    Check that loss accepts if all input arrays are either all float32 or all
    float64, and all output arrays are either all float32 or all float64.

    Also check that input arrays can be readonly, e.g. memory mapped.
    """
    if _IS_WASM and readonly_memmap:
        pytest.xfail(reason='memmap not fully supported')
    loss = loss()
    n_samples = 5
    y_true, raw_prediction = random_y_true_raw_prediction(loss=loss, n_samples=n_samples, y_bound=(-100, 100), raw_bound=(-10, 10), seed=42)
    y_true = y_true.astype(dtype_in)
    raw_prediction = raw_prediction.astype(dtype_in)
    if sample_weight is not None:
        sample_weight = np.array([2.0] * n_samples, dtype=dtype_in)
    if out1 is not None:
        out1 = np.empty_like(y_true, dtype=dtype_out)
    if out2 is not None:
        out2 = np.empty_like(raw_prediction, dtype=dtype_out)
    if readonly_memmap:
        y_true = create_memmap_backed_data(y_true)
        raw_prediction = create_memmap_backed_data(raw_prediction)
        if sample_weight is not None:
            sample_weight = create_memmap_backed_data(sample_weight)
    loss.loss(y_true=y_true, raw_prediction=raw_prediction, sample_weight=sample_weight, loss_out=out1, n_threads=n_threads)
    loss.gradient(y_true=y_true, raw_prediction=raw_prediction, sample_weight=sample_weight, gradient_out=out2, n_threads=n_threads)
    loss.loss_gradient(y_true=y_true, raw_prediction=raw_prediction, sample_weight=sample_weight, loss_out=out1, gradient_out=out2, n_threads=n_threads)
    if out1 is not None and loss.is_multiclass:
        out1 = np.empty_like(raw_prediction, dtype=dtype_out)
    loss.gradient_hessian(y_true=y_true, raw_prediction=raw_prediction, sample_weight=sample_weight, gradient_out=out1, hessian_out=out2, n_threads=n_threads)
    loss(y_true=y_true, raw_prediction=raw_prediction, sample_weight=sample_weight)
    loss.fit_intercept_only(y_true=y_true, sample_weight=sample_weight)
    loss.constant_to_optimal_zero(y_true=y_true, sample_weight=sample_weight)
    if hasattr(loss, 'predict_proba'):
        loss.predict_proba(raw_prediction=raw_prediction)
    if hasattr(loss, 'gradient_proba'):
        loss.gradient_proba(y_true=y_true, raw_prediction=raw_prediction, sample_weight=sample_weight, gradient_out=out1, proba_out=out2, n_threads=n_threads)