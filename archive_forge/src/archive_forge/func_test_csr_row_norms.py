import numpy as np
import pytest
import scipy.sparse as sp
from numpy.random import RandomState
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import linalg
from sklearn.datasets import make_classification
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
from sklearn.utils.sparsefuncs import (
from sklearn.utils.sparsefuncs_fast import (
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_csr_row_norms(dtype):
    X = sp.random(100, 10, format='csr', dtype=dtype, random_state=42)
    scipy_norms = sp.linalg.norm(X, axis=1) ** 2
    norms = csr_row_norms(X)
    assert norms.dtype == dtype
    rtol = 1e-06 if dtype == np.float32 else 1e-07
    assert_allclose(norms, scipy_norms, rtol=rtol)