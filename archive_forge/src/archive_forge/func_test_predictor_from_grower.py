import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
def test_predictor_from_grower():
    n_bins = 256
    X_binned, all_gradients, all_hessians = _make_training_data(n_bins=n_bins)
    grower = TreeGrower(X_binned, all_gradients, all_hessians, n_bins=n_bins, shrinkage=1.0, max_leaf_nodes=3, min_samples_leaf=5)
    grower.grow()
    assert grower.n_nodes == 5
    predictor = grower.make_predictor(binning_thresholds=np.zeros((X_binned.shape[1], n_bins)))
    assert predictor.nodes.shape[0] == 5
    assert predictor.nodes['is_leaf'].sum() == 3
    input_data = np.array([[0, 0], [42, 99], [128, 254], [129, 0], [129, 85], [254, 85], [129, 86], [129, 254], [242, 100]], dtype=np.uint8)
    missing_values_bin_idx = n_bins - 1
    predictions = predictor.predict_binned(input_data, missing_values_bin_idx, n_threads)
    expected_targets = [1, 1, 1, 1, 1, 1, -1, -1, -1]
    assert np.allclose(predictions, expected_targets)
    predictions = predictor.predict_binned(X_binned, missing_values_bin_idx, n_threads)
    assert np.allclose(predictions, -all_gradients)