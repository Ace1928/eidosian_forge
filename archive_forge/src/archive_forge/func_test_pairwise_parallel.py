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
@pytest.mark.parametrize('func, metric, kwds', [(pairwise_distances, 'euclidean', {}), pytest.param(pairwise_distances, minkowski, _minkowski_kwds), pytest.param(pairwise_distances, 'minkowski', _minkowski_kwds), pytest.param(pairwise_distances, wminkowski, _wminkowski_kwds, marks=pytest.mark.skipif(sp_version >= parse_version('1.6.0'), reason='wminkowski is now minkowski and it has been already tested.')), pytest.param(pairwise_distances, 'wminkowski', _wminkowski_kwds, marks=pytest.mark.skipif(sp_version >= parse_version('1.6.0'), reason='wminkowski is now minkowski and it has been already tested.')), (pairwise_kernels, 'polynomial', {'degree': 1}), (pairwise_kernels, callable_rbf_kernel, {'gamma': 0.1})])
@pytest.mark.parametrize('dtype', [np.float64, np.float32, int])
def test_pairwise_parallel(func, metric, kwds, dtype):
    rng = np.random.RandomState(0)
    X = np.array(5 * rng.random_sample((5, 4)), dtype=dtype)
    Y = np.array(5 * rng.random_sample((3, 4)), dtype=dtype)
    S = func(X, metric=metric, n_jobs=1, **kwds)
    S2 = func(X, metric=metric, n_jobs=2, **kwds)
    assert_allclose(S, S2)
    S = func(X, Y, metric=metric, n_jobs=1, **kwds)
    S2 = func(X, Y, metric=metric, n_jobs=2, **kwds)
    assert_allclose(S, S2)