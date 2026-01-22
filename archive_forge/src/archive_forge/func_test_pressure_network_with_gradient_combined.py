import warnings
import pytest
from numpy.testing import (assert_, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
import numpy as np
from numpy import array, float64
from multiprocessing.pool import ThreadPool
from scipy import optimize, linalg
from scipy.special import lambertw
from scipy.optimize._minpack_py import leastsq, curve_fit, fixed_point
from scipy.optimize import OptimizeWarning
from scipy.optimize._minimize import Bounds
def test_pressure_network_with_gradient_combined(self):
    k = np.full(4, 0.5)
    Qtot = 4
    initial_guess = array([2.0, 0.0, 2.0, 0.0])
    final_flows = optimize.root(pressure_network_fun_and_grad, initial_guess, args=(Qtot, k), method='hybr', jac=True).x
    assert_array_almost_equal(final_flows, np.ones(4))