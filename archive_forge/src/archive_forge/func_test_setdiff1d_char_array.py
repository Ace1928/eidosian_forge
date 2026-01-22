import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def test_setdiff1d_char_array(self):
    a = np.array(['a', 'b', 'c'])
    b = np.array(['a', 'b', 's'])
    assert_array_equal(setdiff1d(a, b), np.array(['c']))