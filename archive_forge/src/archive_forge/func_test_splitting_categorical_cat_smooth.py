import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder
from sklearn.ensemble._hist_gradient_boosting.splitting import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._testing import skip_if_32bit
@pytest.mark.parametrize('X_binned, has_missing_values, n_bins_non_missing, ', [([0] * 20, False, 1), ([0] * 9 + [1] * 8, False, 2), ([0] * 12 + [1] * 8, False, 2), ([0] * 9 + [1] * 8 + [9] * 4, True, 2), ([9] * 11, True, 0)])
def test_splitting_categorical_cat_smooth(X_binned, has_missing_values, n_bins_non_missing):
    n_bins = max(X_binned) + 1
    n_samples = len(X_binned)
    X_binned = np.array([X_binned], dtype=X_BINNED_DTYPE).T
    X_binned = np.asfortranarray(X_binned)
    l2_regularization = 0.0
    min_hessian_to_split = 0.001
    min_samples_leaf = 1
    min_gain_to_split = 0.0
    sample_indices = np.arange(n_samples, dtype=np.uint32)
    all_gradients = np.ones(n_samples, dtype=G_H_DTYPE)
    has_missing_values = np.array([has_missing_values], dtype=np.uint8)
    all_hessians = np.ones(1, dtype=G_H_DTYPE)
    sum_gradients = all_gradients.sum()
    sum_hessians = n_samples
    hessians_are_constant = True
    builder = HistogramBuilder(X_binned, n_bins, all_gradients, all_hessians, hessians_are_constant, n_threads)
    n_bins_non_missing = np.array([n_bins_non_missing], dtype=np.uint32)
    monotonic_cst = np.array([MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8)
    is_categorical = np.ones_like(monotonic_cst, dtype=np.uint8)
    missing_values_bin_idx = n_bins - 1
    splitter = Splitter(X_binned, n_bins_non_missing, missing_values_bin_idx, has_missing_values, is_categorical, monotonic_cst, l2_regularization, min_hessian_to_split, min_samples_leaf, min_gain_to_split, hessians_are_constant)
    histograms = builder.compute_histograms_brute(sample_indices)
    value = compute_node_value(sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization)
    split_info = splitter.find_node_split(n_samples, histograms, sum_gradients, sum_hessians, value)
    assert split_info.gain == -1