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
def test_callable_jac_with_args_gh11052(self):
    rng = np.random.default_rng(94253637693657847462)

    def f(x, power):
        return np.sum(np.exp(x ** power))

    def jac(x, power):
        return np.exp(x ** power) * power * x ** (power - 1)
    res1 = dual_annealing(f, args=(2,), bounds=[[0, 1], [0, 1]], seed=rng, minimizer_kwargs=dict(method='L-BFGS-B'))
    res2 = dual_annealing(f, args=(2,), bounds=[[0, 1], [0, 1]], seed=rng, minimizer_kwargs=dict(method='L-BFGS-B', jac=jac))
    assert_allclose(res1.fun, res2.fun, rtol=1e-06)