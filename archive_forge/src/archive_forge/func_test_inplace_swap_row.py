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
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_inplace_swap_row(csc_container, csr_container):
    X = np.array([[0, 3, 0], [2, 4, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=np.float64)
    X_csr = csr_container(X)
    X_csc = csc_container(X)
    swap = linalg.get_blas_funcs(('swap',), (X,))
    swap = swap[0]
    X[0], X[-1] = swap(X[0], X[-1])
    inplace_swap_row(X_csr, 0, -1)
    inplace_swap_row(X_csc, 0, -1)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())
    X[2], X[3] = swap(X[2], X[3])
    inplace_swap_row(X_csr, 2, 3)
    inplace_swap_row(X_csc, 2, 3)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())
    with pytest.raises(TypeError):
        inplace_swap_row(X_csr.tolil())
    X = np.array([[0, 3, 0], [2, 4, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=np.float32)
    X_csr = csr_container(X)
    X_csc = csc_container(X)
    swap = linalg.get_blas_funcs(('swap',), (X,))
    swap = swap[0]
    X[0], X[-1] = swap(X[0], X[-1])
    inplace_swap_row(X_csr, 0, -1)
    inplace_swap_row(X_csc, 0, -1)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())
    X[2], X[3] = swap(X[2], X[3])
    inplace_swap_row(X_csr, 2, 3)
    inplace_swap_row(X_csc, 2, 3)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())
    with pytest.raises(TypeError):
        inplace_swap_row(X_csr.tolil())