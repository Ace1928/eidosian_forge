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
@pytest.mark.parametrize('splu_fun, rtol', [(splu, 1e-07), (spilu, 0.1)])
def test_natural_permc(self, splu_fun, rtol):
    np.random.seed(42)
    n = 500
    p = 0.01
    A = scipy.sparse.random(n, n, p)
    x = np.random.rand(n)
    A += (n + 1) * scipy.sparse.identity(n)
    A_ = csc_matrix(A)
    b = A_ @ x
    lu = splu_fun(A_)
    assert_(np.any(lu.perm_c != np.arange(n)))
    lu = splu_fun(A_, permc_spec='NATURAL')
    assert_array_equal(lu.perm_c, np.arange(n))
    x2 = lu.solve(b)
    assert_allclose(x, x2, rtol=rtol)