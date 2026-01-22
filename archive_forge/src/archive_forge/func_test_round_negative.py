import pytest
from numpy import (
from numpy.testing import (
def test_round_negative(self):
    y = linspace(-1, 3, num=8, dtype=int)
    t = array([-1, -1, 0, 0, 1, 1, 2, 3], dtype=int)
    assert_array_equal(y, t)