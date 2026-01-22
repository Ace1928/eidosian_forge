import numpy as np
import pytest
from scipy import linalg, sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.special import expit
from sklearn.datasets import make_low_rank_matrix, make_sparse_spd_matrix
from sklearn.utils import gen_batches
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils._testing import (
from sklearn.utils.extmath import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_safe_sparse_dot_nd(csr_container):
    rng = np.random.RandomState(0)
    A = rng.random_sample((2, 3, 4, 5, 6))
    B = rng.random_sample((6, 7))
    expected = np.dot(A, B)
    B = csr_container(B)
    actual = safe_sparse_dot(A, B)
    assert_allclose(actual, expected)
    A = rng.random_sample((2, 3))
    B = rng.random_sample((4, 5, 3, 6))
    expected = np.dot(A, B)
    A = csr_container(A)
    actual = safe_sparse_dot(A, B)
    assert_allclose(actual, expected)