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
@pytest.mark.parametrize('method', ('online', 'batch'))
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_lda_score(method, csr_container):
    n_components, X = _build_sparse_array(csr_container)
    lda_1 = LatentDirichletAllocation(n_components=n_components, max_iter=1, learning_method=method, total_samples=100, random_state=0)
    lda_2 = LatentDirichletAllocation(n_components=n_components, max_iter=10, learning_method=method, total_samples=100, random_state=0)
    lda_1.fit_transform(X)
    score_1 = lda_1.score(X)
    lda_2.fit_transform(X)
    score_2 = lda_2.score(X)
    assert score_2 >= score_1