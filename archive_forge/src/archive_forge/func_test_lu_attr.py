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
def test_lu_attr(self):

    def check(dtype, complex_2=False):
        A = self.A.astype(dtype)
        if complex_2:
            A = A + 1j * A.T
        n = A.shape[0]
        lu = splu(A)
        Pc = np.zeros((n, n))
        Pc[np.arange(n), lu.perm_c] = 1
        Pr = np.zeros((n, n))
        Pr[lu.perm_r, np.arange(n)] = 1
        Ad = A.toarray()
        lhs = Pr.dot(Ad).dot(Pc)
        rhs = (lu.L @ lu.U).toarray()
        eps = np.finfo(dtype).eps
        assert_allclose(lhs, rhs, atol=100 * eps)
    check(np.float32)
    check(np.float64)
    check(np.complex64)
    check(np.complex128)
    check(np.complex64, True)
    check(np.complex128, True)