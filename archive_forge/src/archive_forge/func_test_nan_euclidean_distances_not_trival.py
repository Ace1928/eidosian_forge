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
@pytest.mark.parametrize('missing_value', [np.nan, -1])
def test_nan_euclidean_distances_not_trival(missing_value):
    X = np.array([[1.0, missing_value, 3.0, 4.0, 2.0], [missing_value, 4.0, 6.0, 1.0, missing_value], [3.0, missing_value, missing_value, missing_value, 1.0]])
    Y = np.array([[missing_value, 7.0, 7.0, missing_value, 2.0], [missing_value, missing_value, 5.0, 4.0, 7.0], [missing_value, missing_value, missing_value, 4.0, 5.0]])
    D1 = nan_euclidean_distances(X, Y, missing_values=missing_value)
    D2 = nan_euclidean_distances(Y, X, missing_values=missing_value)
    assert_almost_equal(D1, D2.T)
    assert_allclose(nan_euclidean_distances(X[:1], Y[:1], squared=True, missing_values=missing_value), [[5.0 / 2.0 * ((7 - 3) ** 2 + (2 - 2) ** 2)]])
    assert_allclose(nan_euclidean_distances(X[1:2], Y[1:2], squared=False, missing_values=missing_value), [[np.sqrt(5.0 / 2.0 * ((6 - 5) ** 2 + (1 - 4) ** 2))]])
    D3 = nan_euclidean_distances(X, missing_values=missing_value)
    D4 = nan_euclidean_distances(X, X, missing_values=missing_value)
    D5 = nan_euclidean_distances(X, X.copy(), missing_values=missing_value)
    assert_allclose(D3, D4)
    assert_allclose(D4, D5)
    D6 = nan_euclidean_distances(X, Y, copy=True)
    D7 = nan_euclidean_distances(X, Y, copy=False)
    assert_allclose(D6, D7)