from numpy.testing import (assert_, assert_allclose, assert_equal,
import pytest
from platform import python_implementation
import numpy as np
from numpy import zeros, array, allclose
from scipy.linalg import norm
from scipy.sparse import csr_matrix, eye, rand
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg import splu
from scipy.sparse.linalg._isolve import lgmres, gmres
def test_preconditioner(self):
    pc = splu(Am.tocsc())
    M = LinearOperator(matvec=pc.solve, shape=A.shape, dtype=A.dtype)
    x0, count_0 = do_solve()
    x1, count_1 = do_solve(M=M)
    assert_(count_1 == 3)
    assert_(count_1 < count_0 / 2)
    assert_(allclose(x1, x0, rtol=1e-14))