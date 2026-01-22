import pytest
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splin
from numpy.testing import assert_allclose, assert_equal
@pytest.fixture
def same_matrix(sparse_cls, sp_sparse_cls):
    np.random.seed(1234)
    A_dense = np.random.rand(9, 9)
    return (sp_sparse_cls(A_dense), sparse_cls(A_dense))