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
def test_differential_evolution(self):
    solver = DifferentialEvolutionSolver(self.quadratic, [(-2, 2)], maxiter=1, polish=False)
    result = solver.solve()
    assert_equal(result.fun, self.quadratic(result.x))
    solver = DifferentialEvolutionSolver(self.quadratic, [(-2, 2)], maxiter=1, polish=True)
    result = solver.solve()
    assert_equal(result.fun, self.quadratic(result.x))