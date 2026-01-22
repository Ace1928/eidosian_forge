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
def test_L3(self):

    def f(x):
        x = np.hstack(([0], x))
        fun = x[1] ** 2 + x[2] ** 2 + x[1] * x[2] - 14 * x[1] - 16 * x[2] + (x[3] - 10) ** 2 + 4 * (x[4] - 5) ** 2 + (x[5] - 3) ** 2 + 2 * (x[6] - 1) ** 2 + 5 * x[7] ** 2 + 7 * (x[8] - 11) ** 2 + 2 * (x[9] - 10) ** 2 + (x[10] - 7) ** 2 + 45
        return fun
    A = np.zeros((4, 11))
    A[1, [1, 2, 7, 8]] = (-4, -5, 3, -9)
    A[2, [1, 2, 7, 8]] = (-10, 8, 17, -2)
    A[3, [1, 2, 9, 10]] = (8, -2, -5, 2)
    A = A[1:, 1:]
    b = np.array([-105, 0, -12])

    def c1(x):
        x = np.hstack(([0], x))
        return [3 * x[1] - 6 * x[2] - 12 * (x[9] - 8) ** 2 + 7 * x[10], -3 * (x[1] - 2) ** 2 - 4 * (x[2] - 3) ** 2 - 2 * x[3] ** 2 + 7 * x[4] + 120, -x[1] ** 2 - 2 * (x[2] - 2) ** 2 + 2 * x[1] * x[2] - 14 * x[5] + 6 * x[6], -5 * x[1] ** 2 - 8 * x[2] - (x[3] - 6) ** 2 + 2 * x[4] + 40, -0.5 * (x[1] - 8) ** 2 - 2 * (x[2] - 4) ** 2 - 3 * x[5] ** 2 + x[6] + 30]
    L = LinearConstraint(A, b, np.inf)
    N = NonlinearConstraint(c1, 0, np.inf)
    bounds = [(-10, 10)] * 10
    constraints = (L, N)
    with suppress_warnings() as sup:
        sup.filter(UserWarning)
        res = differential_evolution(f, bounds, seed=1234, constraints=constraints, popsize=3)
    x_opt = (2.171996, 2.363683, 8.773926, 5.095984, 0.9906548, 1.430574, 1.321644, 9.828726, 8.280092, 8.375927)
    f_opt = 24.3062091
    assert_allclose(f(x_opt), f_opt, atol=1e-05)
    assert_allclose(res.x, x_opt, atol=1e-06)
    assert_allclose(res.fun, f_opt, atol=1e-05)
    assert res.success
    assert_(np.all(A @ res.x >= b))
    assert_(np.all(np.array(c1(res.x)) >= 0))
    assert_(np.all(res.x >= np.array(bounds)[:, 0]))
    assert_(np.all(res.x <= np.array(bounds)[:, 1]))