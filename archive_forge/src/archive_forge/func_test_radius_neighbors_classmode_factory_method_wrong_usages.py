import itertools
import re
import warnings
from functools import partial
import numpy as np
import pytest
import threadpoolctl
from scipy.spatial.distance import cdist
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.metrics._pairwise_distances_reduction import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_radius_neighbors_classmode_factory_method_wrong_usages():
    rng = np.random.RandomState(1)
    X = rng.rand(100, 10)
    Y = rng.rand(100, 10)
    radius = 5
    metric = 'manhattan'
    weights = 'uniform'
    Y_labels = rng.randint(low=0, high=10, size=100)
    unique_Y_labels = np.unique(Y_labels)
    msg = 'Only float64 or float32 datasets pairs are supported at this time, got: X.dtype=float32 and Y.dtype=float64'
    with pytest.raises(ValueError, match=msg):
        RadiusNeighborsClassMode.compute(X=X.astype(np.float32), Y=Y, radius=radius, metric=metric, weights=weights, Y_labels=Y_labels, unique_Y_labels=unique_Y_labels, outlier_label=None)
    msg = 'Only float64 or float32 datasets pairs are supported at this time, got: X.dtype=float64 and Y.dtype=int32'
    with pytest.raises(ValueError, match=msg):
        RadiusNeighborsClassMode.compute(X=X, Y=Y.astype(np.int32), radius=radius, metric=metric, weights=weights, Y_labels=Y_labels, unique_Y_labels=unique_Y_labels, outlier_label=None)
    with pytest.raises(ValueError, match='radius == -1.0, must be >= 0.'):
        RadiusNeighborsClassMode.compute(X=X, Y=Y, radius=-1, metric=metric, weights=weights, Y_labels=Y_labels, unique_Y_labels=unique_Y_labels, outlier_label=None)
    with pytest.raises(ValueError, match='Unrecognized metric'):
        RadiusNeighborsClassMode.compute(X=X, Y=Y, radius=-1, metric='wrong_metric', weights=weights, Y_labels=Y_labels, unique_Y_labels=unique_Y_labels, outlier_label=None)
    with pytest.raises(ValueError, match='Buffer has wrong number of dimensions \\(expected 2, got 1\\)'):
        RadiusNeighborsClassMode.compute(X=np.array([1.0, 2.0]), Y=Y, radius=radius, metric=metric, weights=weights, Y_labels=Y_labels, unique_Y_labels=unique_Y_labels, outlier_label=None)
    with pytest.raises(ValueError, match='ndarray is not C-contiguous'):
        RadiusNeighborsClassMode.compute(X=np.asfortranarray(X), Y=Y, radius=radius, metric=metric, weights=weights, Y_labels=Y_labels, unique_Y_labels=unique_Y_labels, outlier_label=None)
    non_existent_weights_strategy = 'non_existent_weights_strategy'
    msg = f"Only the 'uniform' or 'distance' weights options are supported at this time. Got: weights='{non_existent_weights_strategy}'."
    with pytest.raises(ValueError, match=msg):
        RadiusNeighborsClassMode.compute(X=X, Y=Y, radius=radius, metric='wrong_metric', weights=non_existent_weights_strategy, Y_labels=Y_labels, unique_Y_labels=unique_Y_labels, outlier_label=None)