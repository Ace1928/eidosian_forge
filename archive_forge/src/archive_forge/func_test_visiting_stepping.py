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
@pytest.mark.parametrize('qv', [1.1, 1.41, 2, 2.62, 2.9])
def test_visiting_stepping(self, qv):
    lu = list(zip(*self.ld_bounds))
    lower = np.array(lu[0])
    upper = np.array(lu[1])
    dim = lower.size
    vd = VisitingDistribution(lower, upper, qv, self.rs)
    values = np.zeros(dim)
    x_step_low = vd.visiting(values, 0, self.high_temperature)
    assert_equal(np.not_equal(x_step_low, 0), True)
    values = np.zeros(dim)
    x_step_high = vd.visiting(values, dim, self.high_temperature)
    assert_equal(np.not_equal(x_step_high[0], 0), True)