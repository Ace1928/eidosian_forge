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
def test_from_docstring(self):

    def func(x):
        return np.sum(x * x - 10 * np.cos(2 * np.pi * x)) + 10 * np.size(x)
    lw = [-5.12] * 10
    up = [5.12] * 10
    ret = dual_annealing(func, bounds=list(zip(lw, up)), seed=1234)
    assert_allclose(ret.x, [-4.26437714e-09, -3.91699361e-09, -1.86149218e-09, -3.9716572e-09, -6.29151648e-09, -6.53145322e-09, -3.93616815e-09, -6.55623025e-09, -6.0577528e-09, -5.00668935e-09], atol=4e-08)
    assert_allclose(ret.fun, 0.0, atol=5e-13)