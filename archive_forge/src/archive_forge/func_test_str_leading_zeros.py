import numpy as np
from numpy.testing import (
import pytest
def test_str_leading_zeros(self):
    p = np.poly1d([4, 3, 2, 1])
    p[3] = 0
    assert_equal(str(p), '   2\n3 x + 2 x + 1')
    p = np.poly1d([1, 2])
    p[0] = 0
    p[1] = 0
    assert_equal(str(p), ' \n0')