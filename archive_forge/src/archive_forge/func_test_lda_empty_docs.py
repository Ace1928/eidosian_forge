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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_lda_empty_docs(csr_container):
    """Test LDA on empty document (all-zero rows)."""
    Z = np.zeros((5, 4))
    for X in [Z, csr_container(Z)]:
        lda = LatentDirichletAllocation(max_iter=750).fit(X)
        assert_almost_equal(lda.components_.sum(axis=0), np.ones(lda.components_.shape[1]))