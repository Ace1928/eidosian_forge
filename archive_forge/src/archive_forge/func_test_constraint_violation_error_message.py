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
def test_constraint_violation_error_message(self):

    def func(x):
        return np.cos(x[0]) + np.sin(x[1])
    c0 = NonlinearConstraint(lambda x: x[1] - (x[0] - 1) ** 2, 0, np.inf)
    c1 = NonlinearConstraint(lambda x: x[1] + x[0] ** 2, -np.inf, 0)
    result = differential_evolution(func, bounds=[(-1, 2), (-1, 1)], constraints=[c0, c1], maxiter=10, polish=False, seed=864197532)
    assert result.success is False
    assert 'MAXCV = 0.414' in result.message