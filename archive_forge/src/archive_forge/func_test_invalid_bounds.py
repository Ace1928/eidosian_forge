from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def test_invalid_bounds(self):
    bounds_list = [((1, 2), (2, 1)), ((2, 1), (1, 2)), ((2, 1), (2, 1)), ((np.inf, 0), (np.inf, 0)), ((1, -np.inf), (0, 1))]
    for bounds in bounds_list:
        with assert_raises(ValueError):
            minimize(self.fun, [-1.0, 1.0], bounds=bounds, method='SLSQP')