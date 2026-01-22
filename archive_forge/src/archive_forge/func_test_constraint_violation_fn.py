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
def test_constraint_violation_fn(self):

    def constr_f(x):
        return [x[0] + x[1]]

    def constr_f2(x):
        return np.array([x[0] ** 2 + x[1], x[0] - x[1]])
    nlc = NonlinearConstraint(constr_f, -np.inf, 1.9)
    solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)], constraints=nlc)
    cv = solver._constraint_violation_fn(np.array([1.0, 1.0]))
    assert_almost_equal(cv, 0.1)
    nlc2 = NonlinearConstraint(constr_f2, -np.inf, 1.8)
    solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)], constraints=(nlc, nlc2))
    xs = [(1.2, 1), (2.0, 2.0), (0.5, 0.5)]
    vs = [(0.3, 0.64, 0.0), (2.1, 4.2, 0.0), (0, 0, 0)]
    for x, v in zip(xs, vs):
        cv = solver._constraint_violation_fn(np.array(x))
        assert_allclose(cv, np.atleast_2d(v))
    assert_allclose(solver._constraint_violation_fn(np.array(xs)), np.array(vs))
    constraint_violation = np.array([solver._constraint_violation_fn(x) for x in np.array(xs)])
    assert constraint_violation.shape == (3, 1, 3)

    def constr_f3(x):
        return constr_f2(x).T
    nlc2 = NonlinearConstraint(constr_f3, -np.inf, 1.8)
    solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)], constraints=(nlc, nlc2), vectorized=False)
    solver.vectorized = True
    with pytest.raises(RuntimeError, match='An array returned from a Constraint'):
        solver._constraint_violation_fn(np.array(xs))