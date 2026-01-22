import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def test_in1d_second_array_is_object(self):
    ar1 = 1
    ar2 = np.array([None] * 10)
    expected = np.array([False])
    result = np.in1d(ar1, ar2)
    assert_array_equal(result, expected)