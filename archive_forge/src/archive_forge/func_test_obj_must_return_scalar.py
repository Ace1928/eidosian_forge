from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def test_obj_must_return_scalar(self):
    with assert_raises(ValueError):
        fmin_slsqp(lambda x: [0, 1], [1, 2, 3])