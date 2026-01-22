import pytest
from numpy import (
from numpy.testing import (
def test_any_step_zero_and_not_mult_inplace(self):
    start = array([0.0, 1.0])
    stop = array([2.0, 1.0])
    y = linspace(start, stop, 3)
    assert_array_equal(y, array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]))