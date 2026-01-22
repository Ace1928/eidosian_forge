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
def test_minimizer_kwargs_bounds(self):

    def func(x):
        return np.sum((x - 5) * (x - 1))
    bounds = list(zip([-6, -5], [6, 5]))
    dual_annealing(func, bounds=bounds, minimizer_kwargs={'method': 'SLSQP', 'bounds': bounds})
    with pytest.warns(RuntimeWarning, match='Method CG cannot handle '):
        dual_annealing(func, bounds=bounds, minimizer_kwargs={'method': 'CG', 'bounds': bounds})