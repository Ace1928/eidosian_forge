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
def test_constraint_wrapper_violation(self):

    def cons_f(x):
        return np.array([x[0] ** 2 + x[1], x[0] ** 2 - x[1]])
    nlc = NonlinearConstraint(cons_f, [-1, -0.85], [2, 2])
    pc = _ConstraintWrapper(nlc, [0.5, 1])
    assert np.size(pc.bounds[0]) == 2
    xs = [(0.5, 1), (0.5, 1.2), (1.2, 1.2), (0.1, -1.2), (0.1, 2.0)]
    vs = [(0, 0), (0, 0.1), (0.64, 0), (0.19, 0), (0.01, 1.14)]
    for x, v in zip(xs, vs):
        assert_allclose(pc.violation(x), v)
    assert_allclose(pc.violation(np.array(xs).T), np.array(vs).T)
    assert pc.fun(np.array(xs).T).shape == (2, len(xs))
    assert pc.violation(np.array(xs).T).shape == (2, len(xs))
    assert pc.num_constr == 2
    assert pc.parameter_count == 2