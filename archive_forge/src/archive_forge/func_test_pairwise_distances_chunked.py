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
@ignore_warnings
def test_pairwise_distances_chunked(global_dtype):
    rng = np.random.RandomState(0)
    X = rng.random_sample((200, 4)).astype(global_dtype, copy=False)
    check_pairwise_distances_chunked(X, None, working_memory=1, metric='euclidean')
    for power in range(-16, 0):
        check_pairwise_distances_chunked(X, None, working_memory=2 ** power, metric='euclidean')
    check_pairwise_distances_chunked(X.tolist(), None, working_memory=1, metric='euclidean')
    Y = rng.random_sample((100, 4)).astype(global_dtype, copy=False)
    check_pairwise_distances_chunked(X, Y, working_memory=1, metric='euclidean')
    check_pairwise_distances_chunked(X.tolist(), Y.tolist(), working_memory=1, metric='euclidean')
    check_pairwise_distances_chunked(X, Y, working_memory=10000, metric='euclidean')
    check_pairwise_distances_chunked(X, Y, working_memory=1, metric='cityblock')
    D = pairwise_distances(X)
    gen = pairwise_distances_chunked(D, working_memory=2 ** (-16), metric='precomputed')
    assert isinstance(gen, GeneratorType)
    assert next(gen) is D
    with pytest.raises(StopIteration):
        next(gen)