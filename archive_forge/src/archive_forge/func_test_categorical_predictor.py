import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression
from sklearn.ensemble._hist_gradient_boosting._bitset import (
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
@pytest.mark.parametrize('bins_go_left, expected_predictions', [([0, 3, 4, 6], [1, 0, 0, 1, 1, 0]), ([0, 1, 2, 6], [1, 1, 1, 0, 0, 0]), ([3, 5, 6], [0, 0, 0, 1, 0, 1])])
def test_categorical_predictor(bins_go_left, expected_predictions):
    X_binned = np.array([[0, 1, 2, 3, 4, 5]], dtype=X_BINNED_DTYPE).T
    categories = np.array([2, 5, 6, 8, 10, 15], dtype=X_DTYPE)
    bins_go_left = np.array(bins_go_left, dtype=X_BINNED_DTYPE)
    nodes = np.zeros(3, dtype=PREDICTOR_RECORD_DTYPE)
    nodes[0]['left'] = 1
    nodes[0]['right'] = 2
    nodes[0]['feature_idx'] = 0
    nodes[0]['is_categorical'] = True
    nodes[0]['missing_go_to_left'] = True
    nodes[1]['is_leaf'] = True
    nodes[1]['value'] = 1
    nodes[2]['is_leaf'] = True
    nodes[2]['value'] = 0
    binned_cat_bitsets = np.zeros((1, 8), dtype=X_BITSET_INNER_DTYPE)
    raw_categorical_bitsets = np.zeros((1, 8), dtype=X_BITSET_INNER_DTYPE)
    for go_left in bins_go_left:
        set_bitset_memoryview(binned_cat_bitsets[0], go_left)
    set_raw_bitset_from_binned_bitset(raw_categorical_bitsets[0], binned_cat_bitsets[0], categories)
    predictor = TreePredictor(nodes, binned_cat_bitsets, raw_categorical_bitsets)
    prediction_binned = predictor.predict_binned(X_binned, missing_values_bin_idx=6, n_threads=n_threads)
    assert_allclose(prediction_binned, expected_predictions)
    known_cat_bitsets = np.zeros((1, 8), dtype=np.uint32)
    known_cat_bitsets[0, 0] = np.sum(2 ** categories, dtype=np.uint32)
    f_idx_map = np.array([0], dtype=np.uint32)
    predictions = predictor.predict(categories.reshape(-1, 1), known_cat_bitsets, f_idx_map, n_threads)
    assert_allclose(predictions, expected_predictions)
    X_binned_missing = np.array([[6]], dtype=X_BINNED_DTYPE).T
    predictions = predictor.predict_binned(X_binned_missing, missing_values_bin_idx=6, n_threads=n_threads)
    assert_allclose(predictions, [1])
    predictions = predictor.predict(np.array([[np.nan, 17]], dtype=X_DTYPE).T, known_cat_bitsets, f_idx_map, n_threads)
    assert_allclose(predictions, [1, 1])