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
def test_constraint_wrapper(self):
    lb = np.array([0, 20, 30])
    ub = np.array([0.5, np.inf, 70])
    x0 = np.array([1, 2, 3])
    pc = _ConstraintWrapper(Bounds(lb, ub), x0)
    assert (pc.violation(x0) > 0).any()
    assert (pc.violation([0.25, 21, 31]) == 0).all()
    xs = np.arange(1, 16).reshape(5, 3)
    violations = []
    for x in xs:
        violations.append(pc.violation(x))
    np.testing.assert_allclose(pc.violation(xs.T), np.array(violations).T)
    x0 = np.array([1, 2, 3, 4])
    A = np.array([[1, 2, 3, 4], [5, 0, 0, 6], [7, 0, 8, 0]])
    pc = _ConstraintWrapper(LinearConstraint(A, -np.inf, 0), x0)
    assert (pc.violation(x0) > 0).any()
    assert (pc.violation([-10, 2, -10, 4]) == 0).all()
    xs = np.arange(1, 29).reshape(7, 4)
    violations = []
    for x in xs:
        violations.append(pc.violation(x))
    np.testing.assert_allclose(pc.violation(xs.T), np.array(violations).T)
    pc = _ConstraintWrapper(LinearConstraint(csr_matrix(A), -np.inf, 0), x0)
    assert (pc.violation(x0) > 0).any()
    assert (pc.violation([-10, 2, -10, 4]) == 0).all()

    def fun(x):
        return A.dot(x)
    nonlinear = NonlinearConstraint(fun, -np.inf, 0)
    pc = _ConstraintWrapper(nonlinear, [-10, 2, -10, 4])
    assert (pc.violation(x0) > 0).any()
    assert (pc.violation([-10, 2, -10, 4]) == 0).all()