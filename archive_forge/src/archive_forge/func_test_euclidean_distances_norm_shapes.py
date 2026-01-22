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
def test_euclidean_distances_norm_shapes():
    rng = np.random.RandomState(0)
    X = rng.random_sample((10, 10))
    Y = rng.random_sample((20, 10))
    X_norm_squared = (X ** 2).sum(axis=1)
    Y_norm_squared = (Y ** 2).sum(axis=1)
    D1 = euclidean_distances(X, Y, X_norm_squared=X_norm_squared, Y_norm_squared=Y_norm_squared)
    D2 = euclidean_distances(X, Y, X_norm_squared=X_norm_squared.reshape(-1, 1), Y_norm_squared=Y_norm_squared.reshape(-1, 1))
    D3 = euclidean_distances(X, Y, X_norm_squared=X_norm_squared.reshape(1, -1), Y_norm_squared=Y_norm_squared.reshape(1, -1))
    assert_allclose(D2, D1)
    assert_allclose(D3, D1)
    with pytest.raises(ValueError, match='Incompatible dimensions for X'):
        euclidean_distances(X, Y, X_norm_squared=X_norm_squared[:5])
    with pytest.raises(ValueError, match='Incompatible dimensions for Y'):
        euclidean_distances(X, Y, Y_norm_squared=Y_norm_squared[:5])