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
def test_vectorized_constraints(self):

    def constr_f(x):
        return np.array([x[0] + x[1]])

    def constr_f2(x):
        return np.array([x[0] ** 2 + x[1], x[0] - x[1]])
    nlc1 = NonlinearConstraint(constr_f, -np.inf, 1.9)
    nlc2 = NonlinearConstraint(constr_f2, (0.9, 0.5), (2.0, 2.0))

    def rosen_vec(x):
        v = 100 * (x[1:] - x[:-1] ** 2.0) ** 2.0
        v += (1 - x[:-1]) ** 2.0
        return np.squeeze(v)
    bounds = [(0, 10), (0, 10)]
    res1 = differential_evolution(rosen, bounds, updating='deferred', seed=1, constraints=[nlc1, nlc2], polish=False)
    res2 = differential_evolution(rosen_vec, bounds, vectorized=True, updating='deferred', seed=1, constraints=[nlc1, nlc2], polish=False)
    assert_allclose(res1.x, res2.x)