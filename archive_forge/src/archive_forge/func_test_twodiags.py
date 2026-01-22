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
@pytest.mark.parametrize('format', ['csc', 'csr'])
@pytest.mark.parametrize('idx_dtype', [np.int32, np.int64])
def test_twodiags(self, format: str, idx_dtype: np.dtype):
    A = spdiags([[1, 2, 3, 4, 5], [6, 5, 8, 9, 10]], [0, 1], 5, 5, format=format)
    b = array([1, 2, 3, 4, 5])
    cond_A = norm(A.toarray(), 2) * norm(inv(A.toarray()), 2)
    for t in ['f', 'd', 'F', 'D']:
        eps = finfo(t).eps
        b = b.astype(t)
        Asp = A.astype(t)
        Asp.indices = Asp.indices.astype(idx_dtype, copy=False)
        Asp.indptr = Asp.indptr.astype(idx_dtype, copy=False)
        x = spsolve(Asp, b)
        assert_(norm(b - Asp @ x) < 10 * cond_A * eps)