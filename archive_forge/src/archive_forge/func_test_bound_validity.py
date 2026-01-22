from scipy.optimize import dual_annealing, Bounds
from scipy.optimize._dual_annealing import EnergyState
from scipy.optimize._dual_annealing import LocalSearchWrapper
from scipy.optimize._dual_annealing import ObjectiveFunWrapper
from scipy.optimize._dual_annealing import StrategyChain
from scipy.optimize._dual_annealing import VisitingDistribution
from scipy.optimize import rosen, rosen_der
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_array_less
from pytest import raises as assert_raises
from scipy._lib._util import check_random_state
def test_bound_validity(self):
    invalid_bounds = [(-5, 5), (-np.inf, 0), (-5, 5)]
    assert_raises(ValueError, dual_annealing, self.func, invalid_bounds)
    invalid_bounds = [(-5, 5), (0, np.inf), (-5, 5)]
    assert_raises(ValueError, dual_annealing, self.func, invalid_bounds)
    invalid_bounds = [(-5, 5), (0, np.nan), (-5, 5)]
    assert_raises(ValueError, dual_annealing, self.func, invalid_bounds)