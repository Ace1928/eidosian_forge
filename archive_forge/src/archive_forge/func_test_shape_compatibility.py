import sys
import threading
import numpy as np
from numpy import array, finfo, arange, eye, all, unique, ones, dot
import numpy.random as random
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
import scipy.linalg
from scipy.linalg import norm, inv
from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
from scipy.sparse.linalg import SuperLU
from scipy.sparse.linalg._dsolve import (spsolve, use_solver, splu, spilu,
import scipy.sparse
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import ComplexWarning
@sup_sparse_efficiency
@pytest.mark.skipif(not has_umfpack, reason='umfpack not available')
def test_shape_compatibility(self):
    use_solver(useUmfpack=True)
    A = csc_matrix([[1.0, 0], [0, 2]])
    bs = [[1, 6], array([1, 6]), [[1], [6]], array([[1], [6]]), csc_matrix([[1], [6]]), csr_matrix([[1], [6]]), dok_matrix([[1], [6]]), bsr_matrix([[1], [6]]), array([[1.0, 2.0, 3.0], [6.0, 8.0, 10.0]]), csc_matrix([[1.0, 2.0, 3.0], [6.0, 8.0, 10.0]]), csr_matrix([[1.0, 2.0, 3.0], [6.0, 8.0, 10.0]]), dok_matrix([[1.0, 2.0, 3.0], [6.0, 8.0, 10.0]]), bsr_matrix([[1.0, 2.0, 3.0], [6.0, 8.0, 10.0]])]
    for b in bs:
        x = np.linalg.solve(A.toarray(), toarray(b))
        for spmattype in [csc_matrix, csr_matrix, dok_matrix, lil_matrix]:
            x1 = spsolve(spmattype(A), b, use_umfpack=True)
            x2 = spsolve(spmattype(A), b, use_umfpack=False)
            if x.ndim == 2 and x.shape[1] == 1:
                x = x.ravel()
            assert_array_almost_equal(toarray(x1), x, err_msg=repr((b, spmattype, 1)))
            assert_array_almost_equal(toarray(x2), x, err_msg=repr((b, spmattype, 2)))
            if issparse(b) and x.ndim > 1:
                assert_(issparse(x1), repr((b, spmattype, 1)))
                assert_(issparse(x2), repr((b, spmattype, 2)))
            else:
                assert_(isinstance(x1, np.ndarray), repr((b, spmattype, 1)))
                assert_(isinstance(x2, np.ndarray), repr((b, spmattype, 2)))
            if x.ndim == 1:
                assert_equal(x1.shape, (A.shape[1],))
                assert_equal(x2.shape, (A.shape[1],))
            else:
                assert_equal(x1.shape, x.shape)
                assert_equal(x2.shape, x.shape)
    A = csc_matrix((3, 3))
    b = csc_matrix((1, 3))
    assert_raises(ValueError, spsolve, A, b)