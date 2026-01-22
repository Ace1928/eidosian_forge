import pytest
import numpy as np
from numpy.linalg import lstsq
from numpy.testing import assert_allclose, assert_equal, assert_
from scipy.sparse import rand, coo_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import lsq_linear
from scipy.optimize._minimize import Bounds
def test_sparse_bounds(self):
    m = 5000
    n = 1000
    A = rand(m, n, random_state=0)
    b = self.rnd.randn(m)
    lb = self.rnd.randn(n)
    ub = lb + 1
    res = lsq_linear(A, b, (lb, ub))
    assert_allclose(res.optimality, 0.0, atol=1e-06)
    res = lsq_linear(A, b, (lb, ub), lsmr_tol=1e-13, lsmr_maxiter=1500)
    assert_allclose(res.optimality, 0.0, atol=1e-06)
    res = lsq_linear(A, b, (lb, ub), lsmr_tol='auto')
    assert_allclose(res.optimality, 0.0, atol=1e-06)