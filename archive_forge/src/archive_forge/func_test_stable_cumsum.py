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
def test_stable_cumsum():
    assert_array_equal(stable_cumsum([1, 2, 3]), np.cumsum([1, 2, 3]))
    r = np.random.RandomState(0).rand(100000)
    with pytest.warns(RuntimeWarning):
        stable_cumsum(r, rtol=0, atol=0)
    A = np.random.RandomState(36).randint(1000, size=(5, 5, 5))
    assert_array_equal(stable_cumsum(A, axis=0), np.cumsum(A, axis=0))
    assert_array_equal(stable_cumsum(A, axis=1), np.cumsum(A, axis=1))
    assert_array_equal(stable_cumsum(A, axis=2), np.cumsum(A, axis=2))