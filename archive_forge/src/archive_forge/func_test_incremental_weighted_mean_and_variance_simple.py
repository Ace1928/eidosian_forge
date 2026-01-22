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
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_incremental_weighted_mean_and_variance_simple(rng, dtype):
    mult = 10
    X = rng.rand(1000, 20).astype(dtype) * mult
    sample_weight = rng.rand(X.shape[0]) * mult
    mean, var, _ = _incremental_mean_and_var(X, 0, 0, 0, sample_weight=sample_weight)
    expected_mean = np.average(X, weights=sample_weight, axis=0)
    expected_var = np.average(X ** 2, weights=sample_weight, axis=0) - expected_mean ** 2
    assert_almost_equal(mean, expected_mean)
    assert_almost_equal(var, expected_var)