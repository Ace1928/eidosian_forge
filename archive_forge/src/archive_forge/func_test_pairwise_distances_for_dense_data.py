import warnings
from types import GeneratorType
import numpy as np
from numpy import linalg
from scipy.sparse import issparse
from scipy.spatial.distance import (
import pytest
from sklearn import config_context
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import (
from sklearn.preprocessing import normalize
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.parallel import Parallel, delayed
def test_pairwise_distances_for_dense_data(global_dtype):
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
    S = pairwise_distances(X, metric='euclidean')
    S2 = euclidean_distances(X)
    assert_allclose(S, S2)
    assert S.dtype == S2.dtype == global_dtype
    Y = rng.random_sample((2, 4)).astype(global_dtype, copy=False)
    S = pairwise_distances(X, Y, metric='euclidean')
    S2 = euclidean_distances(X, Y)
    assert_allclose(S, S2)
    assert S.dtype == S2.dtype == global_dtype
    X_masked = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
    Y_masked = rng.random_sample((2, 4)).astype(global_dtype, copy=False)
    X_masked[0, 0] = np.nan
    Y_masked[0, 0] = np.nan
    S_masked = pairwise_distances(X_masked, Y_masked, metric='nan_euclidean')
    S2_masked = nan_euclidean_distances(X_masked, Y_masked)
    assert_allclose(S_masked, S2_masked)
    assert S_masked.dtype == S2_masked.dtype == global_dtype
    X_tuples = tuple([tuple([v for v in row]) for row in X])
    Y_tuples = tuple([tuple([v for v in row]) for row in Y])
    S2 = pairwise_distances(X_tuples, Y_tuples, metric='euclidean')
    assert_allclose(S, S2)
    assert S.dtype == S2.dtype == global_dtype
    X = rng.random_sample((5, 2)).astype(global_dtype, copy=False)
    X[:, 0] = (X[:, 0] - 0.5) * 2 * np.pi / 2
    X[:, 1] = (X[:, 1] - 0.5) * 2 * np.pi
    S = pairwise_distances(X, metric='haversine')
    S2 = haversine_distances(X)
    assert_allclose(S, S2)
    Y = rng.random_sample((2, 2)).astype(global_dtype, copy=False)
    Y[:, 0] = (Y[:, 0] - 0.5) * 2 * np.pi / 2
    Y[:, 1] = (Y[:, 1] - 0.5) * 2 * np.pi
    S = pairwise_distances(X, Y, metric='haversine')
    S2 = haversine_distances(X, Y)
    assert_allclose(S, S2)
    S = pairwise_distances(X, metric='cityblock')
    S2 = pairwise_distances(X, metric=cityblock)
    assert S.shape[0] == S.shape[1]
    assert S.shape[0] == X.shape[0]
    assert_allclose(S, S2)
    S = pairwise_distances(X, Y, metric='manhattan')
    S2 = pairwise_distances(X, Y, metric=cityblock)
    assert S.shape[0] == X.shape[0]
    assert S.shape[1] == Y.shape[0]
    assert_allclose(S, S2)
    S = pairwise_distances(X, Y, metric='cosine')
    S2 = pairwise_distances(X, Y, metric=cosine)
    assert S.shape[0] == X.shape[0]
    assert S.shape[1] == Y.shape[0]
    assert_allclose(S, S2)