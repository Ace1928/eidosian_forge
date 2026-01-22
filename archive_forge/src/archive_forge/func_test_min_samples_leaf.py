import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
@pytest.mark.parametrize('n_samples, min_samples_leaf, n_bins, constant_hessian, noise', [(11, 10, 7, True, 0), (13, 10, 42, False, 0), (56, 10, 255, True, 0.1), (101, 3, 7, True, 0), (200, 42, 42, False, 0), (300, 55, 255, True, 0.1), (300, 301, 255, True, 0.1)])
def test_min_samples_leaf(n_samples, min_samples_leaf, n_bins, constant_hessian, noise):
    rng = np.random.RandomState(seed=0)
    X = rng.normal(size=(n_samples, 3))
    y = X[:, 0] - X[:, 1]
    if noise:
        y_scale = y.std()
        y += rng.normal(scale=noise, size=n_samples) * y_scale
    mapper = _BinMapper(n_bins=n_bins)
    X = mapper.fit_transform(X)
    all_gradients = y.astype(G_H_DTYPE)
    shape_hessian = 1 if constant_hessian else all_gradients.shape
    all_hessians = np.ones(shape=shape_hessian, dtype=G_H_DTYPE)
    grower = TreeGrower(X, all_gradients, all_hessians, n_bins=n_bins, shrinkage=1.0, min_samples_leaf=min_samples_leaf, max_leaf_nodes=n_samples)
    grower.grow()
    predictor = grower.make_predictor(binning_thresholds=mapper.bin_thresholds_)
    if n_samples >= min_samples_leaf:
        for node in predictor.nodes:
            if node['is_leaf']:
                assert node['count'] >= min_samples_leaf
    else:
        assert predictor.nodes.shape[0] == 1
        assert predictor.nodes[0]['is_leaf']
        assert predictor.nodes[0]['count'] == n_samples