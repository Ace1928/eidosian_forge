import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
def test_missing_value_predict_only():
    rng = np.random.RandomState(0)
    n_samples = 100
    X_binned = rng.randint(0, 256, size=(n_samples, 1), dtype=np.uint8)
    X_binned = np.asfortranarray(X_binned)
    gradients = rng.normal(size=n_samples).astype(G_H_DTYPE)
    hessians = np.ones(shape=1, dtype=G_H_DTYPE)
    grower = TreeGrower(X_binned, gradients, hessians, min_samples_leaf=5, has_missing_values=False)
    grower.grow()
    predictor = grower.make_predictor(binning_thresholds=np.zeros((X_binned.shape[1], X_binned.max() + 1)))
    node = predictor.nodes[0]
    while not node['is_leaf']:
        left = predictor.nodes[node['left']]
        right = predictor.nodes[node['right']]
        node = left if left['count'] > right['count'] else right
    prediction_main_path = node['value']
    all_nans = np.full(shape=(n_samples, 1), fill_value=np.nan)
    known_cat_bitsets = np.zeros((0, 8), dtype=X_BITSET_INNER_DTYPE)
    f_idx_map = np.zeros(0, dtype=np.uint32)
    y_pred = predictor.predict(all_nans, known_cat_bitsets, f_idx_map, n_threads)
    assert np.all(y_pred == prediction_main_path)