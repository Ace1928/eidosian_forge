from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def test_array_bounds(self):
    bounds = [(-np.inf, np.inf), (np.array([2]), np.array([3]))]
    x = fmin_slsqp(lambda z: np.sum(z ** 2 - 1), [2.5, 2.5], bounds=bounds, iprint=0)
    assert_array_almost_equal(x, [0, 2])