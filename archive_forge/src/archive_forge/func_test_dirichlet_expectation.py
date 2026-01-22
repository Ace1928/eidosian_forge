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
def test_dirichlet_expectation():
    """Test Cython version of Dirichlet expectation calculation."""
    x = np.logspace(-100, 10, 10000)
    expectation = np.empty_like(x)
    _dirichlet_expectation_1d(x, 0, expectation)
    assert_allclose(expectation, np.exp(psi(x) - psi(np.sum(x))), atol=1e-19)
    x = x.reshape(100, 100)
    assert_allclose(_dirichlet_expectation_2d(x), psi(x) - psi(np.sum(x, axis=1)[:, np.newaxis]), rtol=1e-11, atol=3e-09)