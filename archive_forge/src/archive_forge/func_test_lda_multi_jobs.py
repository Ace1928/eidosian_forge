import sys
from io import StringIO
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.linalg import block_diag
from scipy.special import psi
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition._online_lda_fast import (
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@if_safe_multiprocessing_with_blas
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
@pytest.mark.parametrize('method', ('online', 'batch'))
def test_lda_multi_jobs(method, csr_container):
    n_components, X = _build_sparse_array(csr_container)
    rng = np.random.RandomState(0)
    lda = LatentDirichletAllocation(n_components=n_components, n_jobs=2, learning_method=method, evaluate_every=1, random_state=rng)
    lda.fit(X)
    correct_idx_grps = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    for c in lda.components_:
        top_idx = set(c.argsort()[-3:][::-1])
        assert tuple(sorted(top_idx)) in correct_idx_grps