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
@pytest.mark.parametrize('qv', [2.25, 2.62, 2.9])
def test_visiting_dist_high_temperature(self, qv):
    lu = list(zip(*self.ld_bounds))
    lower = np.array(lu[0])
    upper = np.array(lu[1])
    vd = VisitingDistribution(lower, upper, qv, self.rs)
    values = vd.visit_fn(self.high_temperature, self.nbtestvalues)
    assert_array_less(np.min(values), 1e-10)
    assert_array_less(10000000000.0, np.max(values))