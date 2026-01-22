import re
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_nystroem_singular_kernel():
    rng = np.random.RandomState(0)
    X = rng.rand(10, 20)
    X = np.vstack([X] * 2)
    gamma = 100
    N = Nystroem(gamma=gamma, n_components=X.shape[0]).fit(X)
    X_transformed = N.transform(X)
    K = rbf_kernel(X, gamma=gamma)
    assert_array_almost_equal(K, np.dot(X_transformed, X_transformed.T))
    assert np.all(np.isfinite(Y))