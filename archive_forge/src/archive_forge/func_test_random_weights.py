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
def test_random_weights():
    mode_result = 6
    rng = np.random.RandomState(0)
    x = rng.randint(mode_result, size=(100, 10))
    w = rng.random_sample(x.shape)
    x[:, :5] = mode_result
    w[:, :5] += 1
    mode, score = weighted_mode(x, w, axis=1)
    assert_array_equal(mode, mode_result)
    assert_array_almost_equal(score.ravel(), w[:, :5].sum(1))