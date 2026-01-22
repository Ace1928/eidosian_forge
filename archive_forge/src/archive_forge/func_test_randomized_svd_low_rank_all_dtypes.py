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
@pytest.mark.parametrize('dtype', (np.int32, np.int64, np.float32, np.float64))
def test_randomized_svd_low_rank_all_dtypes(dtype):
    n_samples = 100
    n_features = 500
    rank = 5
    k = 10
    decimal = 5 if dtype == np.float32 else 7
    dtype = np.dtype(dtype)
    X = make_low_rank_matrix(n_samples=n_samples, n_features=n_features, effective_rank=rank, tail_strength=0.0, random_state=0).astype(dtype, copy=False)
    assert X.shape == (n_samples, n_features)
    U, s, Vt = linalg.svd(X, full_matrices=False)
    U = U.astype(dtype, copy=False)
    s = s.astype(dtype, copy=False)
    Vt = Vt.astype(dtype, copy=False)
    for normalizer in ['auto', 'LU', 'QR']:
        Ua, sa, Va = randomized_svd(X, k, power_iteration_normalizer=normalizer, random_state=0)
        if dtype.kind == 'f':
            assert Ua.dtype == dtype
            assert sa.dtype == dtype
            assert Va.dtype == dtype
        else:
            assert Ua.dtype == np.float64
            assert sa.dtype == np.float64
            assert Va.dtype == np.float64
        assert Ua.shape == (n_samples, k)
        assert sa.shape == (k,)
        assert Va.shape == (k, n_features)
        assert_almost_equal(s[:k], sa, decimal=decimal)
        assert_almost_equal(np.dot(U[:, :k], Vt[:k, :]), np.dot(Ua, Va), decimal=decimal)
        for csr_container in CSR_CONTAINERS:
            X = csr_container(X)
            Ua, sa, Va = randomized_svd(X, k, power_iteration_normalizer=normalizer, random_state=0)
            if dtype.kind == 'f':
                assert Ua.dtype == dtype
                assert sa.dtype == dtype
                assert Va.dtype == dtype
            else:
                assert Ua.dtype.kind == 'f'
                assert sa.dtype.kind == 'f'
                assert Va.dtype.kind == 'f'
            assert_almost_equal(s[:rank], sa[:rank], decimal=decimal)