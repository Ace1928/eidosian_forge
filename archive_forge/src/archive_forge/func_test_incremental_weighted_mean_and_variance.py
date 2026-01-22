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
@pytest.mark.parametrize('mean', [0, 10000000.0, -10000000.0])
@pytest.mark.parametrize('var', [1, 1e-08, 100000.0])
@pytest.mark.parametrize('weight_loc, weight_scale', [(0, 1), (0, 1e-08), (1, 1e-08), (10, 1), (10000000.0, 1)])
def test_incremental_weighted_mean_and_variance(mean, var, weight_loc, weight_scale, rng):

    def _assert(X, sample_weight, expected_mean, expected_var):
        n = X.shape[0]
        for chunk_size in [1, n // 10 + 1, n // 4 + 1, n // 2 + 1, n]:
            last_mean, last_weight_sum, last_var = (0, 0, 0)
            for batch in gen_batches(n, chunk_size):
                last_mean, last_var, last_weight_sum = _incremental_mean_and_var(X[batch], last_mean, last_var, last_weight_sum, sample_weight=sample_weight[batch])
            assert_allclose(last_mean, expected_mean)
            assert_allclose(last_var, expected_var, atol=1e-06)
    size = (100, 20)
    weight = rng.normal(loc=weight_loc, scale=weight_scale, size=size[0])
    X = rng.normal(loc=mean, scale=var, size=size)
    expected_mean = _safe_accumulator_op(np.average, X, weights=weight, axis=0)
    expected_var = _safe_accumulator_op(np.average, (X - expected_mean) ** 2, weights=weight, axis=0)
    _assert(X, weight, expected_mean, expected_var)
    X = rng.normal(loc=mean, scale=var, size=size)
    ones_weight = np.ones(size[0])
    expected_mean = _safe_accumulator_op(np.mean, X, axis=0)
    expected_var = _safe_accumulator_op(np.var, X, axis=0)
    _assert(X, ones_weight, expected_mean, expected_var)