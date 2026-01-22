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
def test_haversine_distances():

    def slow_haversine_distances(x, y):
        diff_lat = y[0] - x[0]
        diff_lon = y[1] - x[1]
        a = np.sin(diff_lat / 2) ** 2 + np.cos(x[0]) * np.cos(y[0]) * np.sin(diff_lon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return c
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 2))
    Y = rng.random_sample((10, 2))
    D1 = np.array([[slow_haversine_distances(x, y) for y in Y] for x in X])
    D2 = haversine_distances(X, Y)
    assert_allclose(D1, D2)
    X = rng.random_sample((10, 3))
    err_msg = 'Haversine distance only valid in 2 dimensions'
    with pytest.raises(ValueError, match=err_msg):
        haversine_distances(X)