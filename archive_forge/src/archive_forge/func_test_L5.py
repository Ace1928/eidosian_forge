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
def test_L5(self):

    def f(x):
        x = np.hstack(([0], x))
        fun = np.sin(2 * np.pi * x[1]) ** 3 * np.sin(2 * np.pi * x[2]) / (x[1] ** 3 * (x[1] + x[2]))
        return -fun

    def c1(x):
        x = np.hstack(([0], x))
        return [x[1] ** 2 - x[2] + 1, 1 - x[1] + (x[2] - 4) ** 2]
    N = NonlinearConstraint(c1, -np.inf, 0)
    bounds = [(0, 10)] * 2
    constraints = N
    res = differential_evolution(f, bounds, strategy='rand1bin', seed=1234, constraints=constraints)
    x_opt = (1.22797135, 4.24537337)
    f_opt = -0.095825
    assert_allclose(f(x_opt), f_opt, atol=2e-05)
    assert_allclose(res.fun, f_opt, atol=0.0001)
    assert res.success
    assert_(np.all(np.array(c1(res.x)) <= 0))
    assert_(np.all(res.x >= np.array(bounds)[:, 0]))
    assert_(np.all(res.x <= np.array(bounds)[:, 1]))