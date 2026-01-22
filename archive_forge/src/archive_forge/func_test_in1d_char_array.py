import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def test_in1d_char_array(self):
    a = np.array(['a', 'b', 'c', 'd', 'e', 'c', 'e', 'b'])
    b = np.array(['a', 'c'])
    ec = np.array([True, False, True, False, False, True, False, False])
    c = in1d(a, b)
    assert_array_equal(c, ec)