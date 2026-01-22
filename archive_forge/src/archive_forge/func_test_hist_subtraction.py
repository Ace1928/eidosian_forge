import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.histogram import (
@pytest.mark.parametrize('constant_hessian', [True, False])
def test_hist_subtraction(constant_hessian):
    rng = np.random.RandomState(42)
    n_samples = 10
    n_bins = 5
    sample_indices = np.arange(n_samples).astype(np.uint32)
    binned_feature = rng.randint(0, n_bins - 1, size=n_samples, dtype=np.uint8)
    ordered_gradients = rng.randn(n_samples).astype(G_H_DTYPE)
    if constant_hessian:
        ordered_hessians = np.ones(n_samples, dtype=G_H_DTYPE)
    else:
        ordered_hessians = rng.lognormal(size=n_samples).astype(G_H_DTYPE)
    hist_parent = np.zeros((1, n_bins), dtype=HISTOGRAM_DTYPE)
    if constant_hessian:
        _build_histogram_no_hessian(0, sample_indices, binned_feature, ordered_gradients, hist_parent)
    else:
        _build_histogram(0, sample_indices, binned_feature, ordered_gradients, ordered_hessians, hist_parent)
    mask = rng.randint(0, 2, n_samples).astype(bool)
    sample_indices_left = sample_indices[mask]
    ordered_gradients_left = ordered_gradients[mask]
    ordered_hessians_left = ordered_hessians[mask]
    hist_left = np.zeros((1, n_bins), dtype=HISTOGRAM_DTYPE)
    if constant_hessian:
        _build_histogram_no_hessian(0, sample_indices_left, binned_feature, ordered_gradients_left, hist_left)
    else:
        _build_histogram(0, sample_indices_left, binned_feature, ordered_gradients_left, ordered_hessians_left, hist_left)
    sample_indices_right = sample_indices[~mask]
    ordered_gradients_right = ordered_gradients[~mask]
    ordered_hessians_right = ordered_hessians[~mask]
    hist_right = np.zeros((1, n_bins), dtype=HISTOGRAM_DTYPE)
    if constant_hessian:
        _build_histogram_no_hessian(0, sample_indices_right, binned_feature, ordered_gradients_right, hist_right)
    else:
        _build_histogram(0, sample_indices_right, binned_feature, ordered_gradients_right, ordered_hessians_right, hist_right)
    hist_left_sub = np.copy(hist_parent)
    hist_right_sub = np.copy(hist_parent)
    _subtract_histograms(0, n_bins, hist_left_sub, hist_right)
    _subtract_histograms(0, n_bins, hist_right_sub, hist_left)
    for key in ('count', 'sum_hessians', 'sum_gradients'):
        assert_allclose(hist_left[key], hist_left_sub[key], rtol=1e-06)
        assert_allclose(hist_right[key], hist_right_sub[key], rtol=1e-06)