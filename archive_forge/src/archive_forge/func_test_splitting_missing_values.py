import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder
from sklearn.ensemble._hist_gradient_boosting.splitting import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._testing import skip_if_32bit
@pytest.mark.parametrize('X_binned, all_gradients, has_missing_values, n_bins_non_missing,  expected_split_on_nan, expected_bin_idx, expected_go_to_left', [([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 1, 1, 1, 5, 5, 5, 5, 5, 5], False, 10, False, 3, 'not_applicable'), ([8, 0, 1, 8, 2, 3, 4, 5, 6, 7], [1, 1, 1, 1, 5, 5, 5, 5, 5, 5], True, 8, False, 1, True), ([9, 0, 1, 9, 2, 3, 4, 5, 6, 7], [1, 1, 1, 1, 5, 5, 5, 5, 5, 5], True, 8, False, 1, True), ([0, 1, 2, 3, 8, 4, 8, 5, 6, 7], [1, 1, 1, 1, 5, 5, 5, 5, 5, 5], True, 8, False, 3, False), ([0, 1, 2, 3, 9, 4, 9, 5, 6, 7], [1, 1, 1, 1, 5, 5, 5, 5, 5, 5], True, 8, False, 3, False), ([0, 1, 2, 3, 4, 4, 4, 4, 4, 4], [1, 1, 1, 1, 5, 5, 5, 5, 5, 5], True, 4, True, 3, False), ([0, 1, 2, 3, 9, 9, 9, 9, 9, 9], [1, 1, 1, 1, 1, 1, 5, 5, 5, 5], True, 4, True, 3, False), ([6, 6, 6, 6, 0, 1, 2, 3, 4, 5], [1, 1, 1, 1, 5, 5, 5, 5, 5, 5], True, 6, True, 5, False), ([9, 9, 9, 9, 0, 1, 2, 3, 4, 5], [1, 1, 1, 1, 5, 5, 5, 5, 5, 5], True, 6, True, 5, False)])
def test_splitting_missing_values(X_binned, all_gradients, has_missing_values, n_bins_non_missing, expected_split_on_nan, expected_bin_idx, expected_go_to_left):
    n_bins = max(X_binned) + 1
    n_samples = len(X_binned)
    l2_regularization = 0.0
    min_hessian_to_split = 0.001
    min_samples_leaf = 1
    min_gain_to_split = 0.0
    sample_indices = np.arange(n_samples, dtype=np.uint32)
    X_binned = np.array(X_binned, dtype=X_BINNED_DTYPE).reshape(-1, 1)
    X_binned = np.asfortranarray(X_binned)
    all_gradients = np.array(all_gradients, dtype=G_H_DTYPE)
    has_missing_values = np.array([has_missing_values], dtype=np.uint8)
    all_hessians = np.ones(1, dtype=G_H_DTYPE)
    sum_gradients = all_gradients.sum()
    sum_hessians = 1 * n_samples
    hessians_are_constant = True
    builder = HistogramBuilder(X_binned, n_bins, all_gradients, all_hessians, hessians_are_constant, n_threads)
    n_bins_non_missing = np.array([n_bins_non_missing], dtype=np.uint32)
    monotonic_cst = np.array([MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8)
    is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)
    missing_values_bin_idx = n_bins - 1
    splitter = Splitter(X_binned, n_bins_non_missing, missing_values_bin_idx, has_missing_values, is_categorical, monotonic_cst, l2_regularization, min_hessian_to_split, min_samples_leaf, min_gain_to_split, hessians_are_constant)
    histograms = builder.compute_histograms_brute(sample_indices)
    value = compute_node_value(sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization)
    split_info = splitter.find_node_split(n_samples, histograms, sum_gradients, sum_hessians, value)
    assert split_info.bin_idx == expected_bin_idx
    if has_missing_values:
        assert split_info.missing_go_to_left == expected_go_to_left
    split_on_nan = split_info.bin_idx == n_bins_non_missing[0] - 1
    assert split_on_nan == expected_split_on_nan
    samples_left, samples_right, _ = splitter.split_indices(split_info, splitter.partition)
    if not expected_split_on_nan:
        assert set(samples_left) == set([0, 1, 2, 3])
        assert set(samples_right) == set([4, 5, 6, 7, 8, 9])
    else:
        missing_samples_indices = np.flatnonzero(np.array(X_binned) == missing_values_bin_idx)
        non_missing_samples_indices = np.flatnonzero(np.array(X_binned) != missing_values_bin_idx)
        assert set(samples_right) == set(missing_samples_indices)
        assert set(samples_left) == set(non_missing_samples_indices)