import numpy as np
import pytest
import scipy.sparse as sp
from numpy.random import RandomState
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import linalg
from sklearn.datasets import make_classification
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
from sklearn.utils.sparsefuncs import (
from sklearn.utils.sparsefuncs_fast import (
@pytest.fixture(scope='module', params=CSR_CONTAINERS + CSC_CONTAINERS)
def centered_matrices(request):
    """Returns equivalent tuple[sp.linalg.LinearOperator, np.ndarray]."""
    sparse_container = request.param
    random_state = np.random.default_rng(42)
    X_sparse = sparse_container(sp.random(500, 100, density=0.1, format='csr', random_state=random_state))
    X_dense = X_sparse.toarray()
    mu = np.asarray(X_sparse.mean(axis=0)).ravel()
    X_sparse_centered = _implicit_column_offset(X_sparse, mu)
    X_dense_centered = X_dense - mu
    return (X_sparse_centered, X_dense_centered)