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
def test_incremental_variance_update_formulas():
    A = np.array([[600, 470, 170, 430, 300], [600, 470, 170, 430, 300], [600, 470, 170, 430, 300], [600, 470, 170, 430, 300]]).T
    idx = 2
    X1 = A[:idx, :]
    X2 = A[idx:, :]
    old_means = X1.mean(axis=0)
    old_variances = X1.var(axis=0)
    old_sample_count = np.full(X1.shape[1], X1.shape[0], dtype=np.int32)
    final_means, final_variances, final_count = _incremental_mean_and_var(X2, old_means, old_variances, old_sample_count)
    assert_almost_equal(final_means, A.mean(axis=0), 6)
    assert_almost_equal(final_variances, A.var(axis=0), 6)
    assert_almost_equal(final_count, A.shape[0])