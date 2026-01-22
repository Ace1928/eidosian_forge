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
@pytest.mark.slow
@pytest.mark.xfail(platform.machine() == 'ppc64le', reason='fails on ppc64le')
def test_L8(self):

    def f(x):
        x = np.hstack(([0], x))
        fun = 3 * x[1] + 1e-06 * x[1] ** 3 + 2 * x[2] + 2e-06 / 3 * x[2] ** 3
        return fun
    A = np.zeros((3, 5))
    A[1, [4, 3]] = (1, -1)
    A[2, [3, 4]] = (1, -1)
    A = A[1:, 1:]
    b = np.array([-0.55, -0.55])

    def c1(x):
        x = np.hstack(([0], x))
        return [1000 * np.sin(-x[3] - 0.25) + 1000 * np.sin(-x[4] - 0.25) + 894.8 - x[1], 1000 * np.sin(x[3] - 0.25) + 1000 * np.sin(x[3] - x[4] - 0.25) + 894.8 - x[2], 1000 * np.sin(x[4] - 0.25) + 1000 * np.sin(x[4] - x[3] - 0.25) + 1294.8]
    L = LinearConstraint(A, b, np.inf)
    N = NonlinearConstraint(c1, np.full(3, -0.001), np.full(3, 0.001))
    bounds = [(0, 1200)] * 2 + [(-0.55, 0.55)] * 2
    constraints = (L, N)
    with suppress_warnings() as sup:
        sup.filter(UserWarning)
        res = differential_evolution(f, bounds, strategy='best1bin', seed=1234, constraints=constraints, maxiter=5000)
    x_opt = (679.9453, 1026.067, 0.1188764, -0.3962336)
    f_opt = 5126.4981
    assert_allclose(f(x_opt), f_opt, atol=0.001)
    assert_allclose(res.x[:2], x_opt[:2], atol=0.002)
    assert_allclose(res.x[2:], x_opt[2:], atol=0.002)
    assert_allclose(res.fun, f_opt, atol=0.02)
    assert res.success
    assert_(np.all(A @ res.x >= b))
    assert_(np.all(np.array(c1(res.x)) >= -0.001))
    assert_(np.all(np.array(c1(res.x)) <= 0.001))
    assert_(np.all(res.x >= np.array(bounds)[:, 0]))
    assert_(np.all(res.x <= np.array(bounds)[:, 1]))