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
def test_argkmin_classmode_strategy_consistent():
    rng = np.random.RandomState(1)
    X = rng.rand(100, 10)
    Y = rng.rand(100, 10)
    k = 5
    metric = 'manhattan'
    weights = 'uniform'
    Y_labels = rng.randint(low=0, high=10, size=100)
    unique_Y_labels = np.unique(Y_labels)
    results_X = ArgKminClassMode.compute(X=X, Y=Y, k=k, metric=metric, weights=weights, Y_labels=Y_labels, unique_Y_labels=unique_Y_labels, strategy='parallel_on_X')
    results_Y = ArgKminClassMode.compute(X=X, Y=Y, k=k, metric=metric, weights=weights, Y_labels=Y_labels, unique_Y_labels=unique_Y_labels, strategy='parallel_on_Y')
    assert_array_equal(results_X, results_Y)