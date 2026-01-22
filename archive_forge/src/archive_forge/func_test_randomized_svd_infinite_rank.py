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
def test_randomized_svd_infinite_rank():
    n_samples = 100
    n_features = 500
    rank = 5
    k = 10
    X = make_low_rank_matrix(n_samples=n_samples, n_features=n_features, effective_rank=rank, tail_strength=1.0, random_state=0)
    assert X.shape == (n_samples, n_features)
    _, s, _ = linalg.svd(X, full_matrices=False)
    for normalizer in ['auto', 'none', 'LU', 'QR']:
        _, sa, _ = randomized_svd(X, k, n_iter=0, power_iteration_normalizer=normalizer, random_state=0)
        assert np.abs(s[:k] - sa).max() > 0.1
        _, sap, _ = randomized_svd(X, k, n_iter=5, power_iteration_normalizer=normalizer, random_state=0)
        assert_almost_equal(s[:k], sap, decimal=3)