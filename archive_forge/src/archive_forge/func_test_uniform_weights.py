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
def test_uniform_weights():
    rng = np.random.RandomState(0)
    x = rng.randint(10, size=(10, 5))
    weights = np.ones(x.shape)
    for axis in (None, 0, 1):
        mode, score = _mode(x, axis)
        mode2, score2 = weighted_mode(x, weights, axis=axis)
        assert_array_equal(mode, mode2)
        assert_array_equal(score, score2)