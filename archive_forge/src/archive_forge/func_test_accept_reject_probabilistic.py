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
@pytest.mark.parametrize('new_e, temp_step, accepted, accept_rate', [(0, 100, 1000, 1.0097587941791923), (0, 2, 1000, 1.2599210498948732), (10, 100, 878, 0.8786035869128718), (10, 60, 695, 0.6812920690579612), (2, 100, 990, 0.9897404249173424)])
def test_accept_reject_probabilistic(self, new_e, temp_step, accepted, accept_rate):
    rs = check_random_state(123)
    count_accepted = 0
    iterations = 1000
    accept_param = -5
    current_energy = 1
    for _ in range(iterations):
        energy_state = EnergyState(lower=None, upper=None)
        energy_state.update_current(current_energy, [0])
        chain = StrategyChain(accept_param, None, None, None, rs, energy_state)
        chain.temperature_step = temp_step
        chain.accept_reject(j=1, e=new_e, x_visit=[2])
        if energy_state.current_energy == new_e:
            count_accepted += 1
    assert count_accepted == accepted
    pqv = 1 - (1 - accept_param) * (new_e - current_energy) / temp_step
    rate = 0 if pqv <= 0 else np.exp(np.log(pqv) / (1 - accept_param))
    assert_allclose(rate, accept_rate)