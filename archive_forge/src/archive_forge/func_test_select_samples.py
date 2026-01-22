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
def test_select_samples(self):
    limits = np.arange(12.0, dtype='float64').reshape(2, 6)
    bounds = list(zip(limits[0, :], limits[1, :]))
    solver = DifferentialEvolutionSolver(None, bounds, popsize=1)
    candidate = 0
    r1, r2, r3, r4, r5 = solver._select_samples(candidate, 5)
    assert_equal(len(np.unique(np.array([candidate, r1, r2, r3, r4, r5]))), 6)