import numpy as np
from numpy.testing import (
import pytest
def test_poly_eq(self):
    p = np.poly1d([1, 2, 3])
    p2 = np.poly1d([1, 2, 4])
    assert_equal(p == None, False)
    assert_equal(p != None, True)
    assert_equal(p == p, True)
    assert_equal(p == p2, False)
    assert_equal(p != p2, True)