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
def test_immediate_updating(self):
    bounds = [(0.0, 2.0), (0.0, 2.0)]
    solver = DifferentialEvolutionSolver(rosen, bounds)
    assert_(solver._updating == 'immediate')
    with warns(UserWarning):
        with DifferentialEvolutionSolver(rosen, bounds, workers=2) as solver:
            pass
    assert_(solver._updating == 'deferred')