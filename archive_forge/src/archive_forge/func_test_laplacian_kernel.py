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
def test_laplacian_kernel():
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    K = laplacian_kernel(X, X)
    assert_allclose(np.diag(K), np.ones(5))
    assert np.all(K > 0)
    assert np.all(K - np.diag(np.diag(K)) < 1)