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
def test_random_generator(self):
    rng = np.random.default_rng()
    inits = ['random', 'latinhypercube', 'sobol', 'halton']
    for init in inits:
        differential_evolution(self.quadratic, [(-100, 100)], polish=False, seed=rng, tol=0.5, init=init)