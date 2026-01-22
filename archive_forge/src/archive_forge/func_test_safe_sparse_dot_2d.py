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
@pytest.mark.parametrize('A_container', [np.array, *CSR_CONTAINERS], ids=['dense'] + [container.__name__ for container in CSR_CONTAINERS])
@pytest.mark.parametrize('B_container', [np.array, *CSR_CONTAINERS], ids=['dense'] + [container.__name__ for container in CSR_CONTAINERS])
def test_safe_sparse_dot_2d(A_container, B_container):
    rng = np.random.RandomState(0)
    A = rng.random_sample((30, 10))
    B = rng.random_sample((10, 20))
    expected = np.dot(A, B)
    A = A_container(A)
    B = B_container(B)
    actual = safe_sparse_dot(A, B, dense_output=True)
    assert_allclose(actual, expected)