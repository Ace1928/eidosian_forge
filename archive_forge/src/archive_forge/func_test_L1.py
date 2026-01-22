import multiprocessing
import platform
from scipy.optimize._differentialevolution import (DifferentialEvolutionSolver,
from scipy.optimize import differential_evolution, OptimizeResult
from scipy.optimize._constraints import (Bounds, NonlinearConstraint,
from scipy.optimize import rosen, minimize
from scipy.sparse import csr_matrix
from scipy import stats
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_almost_equal,
from pytest import raises as assert_raises, warns
import pytest
def test_L1(self):

    def f(x):
        x = np.hstack(([0], x))
        fun = np.sum(5 * x[1:5]) - 5 * x[1:5] @ x[1:5] - np.sum(x[5:])
        return fun
    A = np.zeros((10, 14))
    A[1, [1, 2, 10, 11]] = (2, 2, 1, 1)
    A[2, [1, 10]] = (-8, 1)
    A[3, [4, 5, 10]] = (-2, -1, 1)
    A[4, [1, 3, 10, 11]] = (2, 2, 1, 1)
    A[5, [2, 11]] = (-8, 1)
    A[6, [6, 7, 11]] = (-2, -1, 1)
    A[7, [2, 3, 11, 12]] = (2, 2, 1, 1)
    A[8, [3, 12]] = (-8, 1)
    A[9, [8, 9, 12]] = (-2, -1, 1)
    A = A[1:, 1:]
    b = np.array([10, 0, 0, 10, 0, 0, 10, 0, 0])
    L = LinearConstraint(A, -np.inf, b)
    bounds = [(0, 1)] * 9 + [(0, 100)] * 3 + [(0, 1)]
    res = differential_evolution(f, bounds, strategy='best1bin', seed=1234, constraints=L, popsize=2)
    x_opt = (1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1)
    f_opt = -15
    assert_allclose(f(x_opt), f_opt, atol=0.0006)
    assert res.success
    assert_allclose(res.x, x_opt, atol=0.0006)
    assert_allclose(res.fun, f_opt, atol=0.005)
    assert_(np.all(A @ res.x <= b))
    assert_(np.all(res.x >= np.array(bounds)[:, 0]))
    assert_(np.all(res.x <= np.array(bounds)[:, 1]))
    L = LinearConstraint(csr_matrix(A), -np.inf, b)
    res = differential_evolution(f, bounds, strategy='best1bin', seed=1234, constraints=L, popsize=2)
    assert_allclose(f(x_opt), f_opt)
    assert res.success
    assert_allclose(res.x, x_opt, atol=0.0005)
    assert_allclose(res.fun, f_opt, atol=0.005)
    assert_(np.all(A @ res.x <= b))
    assert_(np.all(res.x >= np.array(bounds)[:, 0]))
    assert_(np.all(res.x <= np.array(bounds)[:, 1]))

    def c1(x):
        x = np.hstack(([0], x))
        return [2 * x[2] + 2 * x[3] + x[11] + x[12], -8 * x[3] + x[12]]

    def c2(x):
        x = np.hstack(([0], x))
        return -2 * x[8] - x[9] + x[12]
    L = LinearConstraint(A[:5, :], -np.inf, b[:5])
    L2 = LinearConstraint(A[5:6, :], -np.inf, b[5:6])
    N = NonlinearConstraint(c1, -np.inf, b[6:8])
    N2 = NonlinearConstraint(c2, -np.inf, b[8:9])
    constraints = (L, N, L2, N2)
    with suppress_warnings() as sup:
        sup.filter(UserWarning)
        res = differential_evolution(f, bounds, strategy='rand1bin', seed=1234, constraints=constraints, popsize=2)
    assert_allclose(res.x, x_opt, atol=0.0006)
    assert_allclose(res.fun, f_opt, atol=0.005)
    assert_(np.all(A @ res.x <= b))
    assert_(np.all(res.x >= np.array(bounds)[:, 0]))
    assert_(np.all(res.x <= np.array(bounds)[:, 1]))