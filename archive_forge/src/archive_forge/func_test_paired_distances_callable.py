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
def test_paired_distances_callable(global_dtype):
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
    Y = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
    S = paired_distances(X, Y, metric='manhattan')
    S2 = paired_distances(X, Y, metric=lambda x, y: np.abs(x - y).sum(axis=0))
    assert_allclose(S, S2)
    Y = rng.random_sample((3, 4))
    with pytest.raises(ValueError):
        paired_distances(X, Y)