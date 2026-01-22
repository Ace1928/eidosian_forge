import pytest
import numpy as np
from numpy.linalg import lstsq
from numpy.testing import assert_allclose, assert_equal, assert_
from scipy.sparse import rand, coo_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import lsq_linear
from scipy.optimize._minimize import Bounds
def test_dense_bounds(self):
    lb = np.array([-1, -10])
    ub = np.array([1, 0])
    unbounded_sol = lstsq(A, b, rcond=-1)[0]
    for lsq_solver in self.lsq_solvers:
        res = lsq_linear(A, b, (lb, ub), method=self.method, lsq_solver=lsq_solver)
        assert_allclose(res.x, lstsq(A, b, rcond=-1)[0])
        assert_allclose(res.unbounded_sol[0], unbounded_sol)
    lb = np.array([0.0, -np.inf])
    for lsq_solver in self.lsq_solvers:
        res = lsq_linear(A, b, (lb, np.inf), method=self.method, lsq_solver=lsq_solver)
        assert_allclose(res.x, np.array([0.0, -4.084174437334673]), atol=1e-06)
        assert_allclose(res.unbounded_sol[0], unbounded_sol)
    lb = np.array([-1, 0])
    for lsq_solver in self.lsq_solvers:
        res = lsq_linear(A, b, (lb, np.inf), method=self.method, lsq_solver=lsq_solver)
        assert_allclose(res.x, np.array([0.448427311733504, 0]), atol=1e-15)
        assert_allclose(res.unbounded_sol[0], unbounded_sol)
    ub = np.array([np.inf, -5])
    for lsq_solver in self.lsq_solvers:
        res = lsq_linear(A, b, (-np.inf, ub), method=self.method, lsq_solver=lsq_solver)
        assert_allclose(res.x, np.array([-0.105560998682388, -5]))
        assert_allclose(res.unbounded_sol[0], unbounded_sol)
    ub = np.array([-1, np.inf])
    for lsq_solver in self.lsq_solvers:
        res = lsq_linear(A, b, (-np.inf, ub), method=self.method, lsq_solver=lsq_solver)
        assert_allclose(res.x, np.array([-1, -4.181102129483254]))
        assert_allclose(res.unbounded_sol[0], unbounded_sol)
    lb = np.array([0, -4])
    ub = np.array([1, 0])
    for lsq_solver in self.lsq_solvers:
        res = lsq_linear(A, b, (lb, ub), method=self.method, lsq_solver=lsq_solver)
        assert_allclose(res.x, np.array([0.005236663400791, -4]))
        assert_allclose(res.unbounded_sol[0], unbounded_sol)