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
@pytest.mark.parametrize('dense_output', [True, False])
def test_safe_sparse_dot_dense_output(dense_output):
    rng = np.random.RandomState(0)
    A = sparse.random(30, 10, density=0.1, random_state=rng)
    B = sparse.random(10, 20, density=0.1, random_state=rng)
    expected = A.dot(B)
    actual = safe_sparse_dot(A, B, dense_output=dense_output)
    assert sparse.issparse(actual) == (not dense_output)
    if dense_output:
        expected = expected.toarray()
    assert_allclose_dense_sparse(actual, expected)