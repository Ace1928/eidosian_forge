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
@pytest.mark.parametrize('sparse_container', DOK_CONTAINERS + LIL_CONTAINERS)
def test_randomized_svd_sparse_warnings(sparse_container):
    rng = np.random.RandomState(42)
    X = make_low_rank_matrix(50, 20, effective_rank=10, random_state=rng)
    n_components = 5
    X = sparse_container(X)
    warn_msg = 'Calculating SVD of a {} is expensive. csr_matrix is more efficient.'.format(sparse_container.__name__)
    with pytest.warns(sparse.SparseEfficiencyWarning, match=warn_msg):
        randomized_svd(X, n_components, n_iter=1, power_iteration_normalizer='none')