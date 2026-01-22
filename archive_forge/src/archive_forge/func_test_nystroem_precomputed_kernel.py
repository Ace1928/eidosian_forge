import re
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_nystroem_precomputed_kernel():
    rnd = np.random.RandomState(12)
    X = rnd.uniform(size=(10, 4))
    K = polynomial_kernel(X, degree=2, coef0=0.1)
    nystroem = Nystroem(kernel='precomputed', n_components=X.shape[0])
    X_transformed = nystroem.fit_transform(K)
    assert_array_almost_equal(np.dot(X_transformed, X_transformed.T), K)
    msg = "Don't pass gamma, coef0 or degree to Nystroem"
    params = ({'gamma': 1}, {'coef0': 1}, {'degree': 2})
    for param in params:
        ny = Nystroem(kernel='precomputed', n_components=X.shape[0], **param)
        with pytest.raises(ValueError, match=msg):
            ny.fit(K)