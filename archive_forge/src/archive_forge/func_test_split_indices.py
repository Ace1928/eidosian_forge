import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder
from sklearn.ensemble._hist_gradient_boosting.splitting import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._testing import skip_if_32bit
def test_split_indices():
    rng = np.random.RandomState(421)
    n_bins = 5
    n_samples = 10
    l2_regularization = 0.0
    min_hessian_to_split = 0.001
    min_samples_leaf = 1
    min_gain_to_split = 0.0
    X_binned = [[0, 0], [0, 3], [0, 4], [0, 0], [0, 0], [0, 0], [0, 0], [0, 4], [0, 0], [0, 4]]
    X_binned = np.asfortranarray(X_binned, dtype=X_BINNED_DTYPE)
    sample_indices = np.arange(n_samples, dtype=np.uint32)
    all_gradients = rng.randn(n_samples).astype(G_H_DTYPE)
    all_hessians = np.ones(1, dtype=G_H_DTYPE)
    sum_gradients = all_gradients.sum()
    sum_hessians = 1 * n_samples
    hessians_are_constant = True
    builder = HistogramBuilder(X_binned, n_bins, all_gradients, all_hessians, hessians_are_constant, n_threads)
    n_bins_non_missing = np.array([n_bins] * X_binned.shape[1], dtype=np.uint32)
    has_missing_values = np.array([False] * X_binned.shape[1], dtype=np.uint8)
    monotonic_cst = np.array([MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8)
    is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)
    missing_values_bin_idx = n_bins - 1
    splitter = Splitter(X_binned, n_bins_non_missing, missing_values_bin_idx, has_missing_values, is_categorical, monotonic_cst, l2_regularization, min_hessian_to_split, min_samples_leaf, min_gain_to_split, hessians_are_constant)
    assert np.all(sample_indices == splitter.partition)
    histograms = builder.compute_histograms_brute(sample_indices)
    value = compute_node_value(sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization)
    si_root = splitter.find_node_split(n_samples, histograms, sum_gradients, sum_hessians, value)
    assert si_root.feature_idx == 1
    assert si_root.bin_idx == 3
    samples_left, samples_right, position_right = splitter.split_indices(si_root, splitter.partition)
    assert set(samples_left) == set([0, 1, 3, 4, 5, 6, 8])
    assert set(samples_right) == set([2, 7, 9])
    assert list(samples_left) == list(splitter.partition[:position_right])
    assert list(samples_right) == list(splitter.partition[position_right:])
    assert samples_left.shape[0] == si_root.n_samples_left
    assert samples_right.shape[0] == si_root.n_samples_right