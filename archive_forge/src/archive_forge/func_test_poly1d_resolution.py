import numpy as np
from numpy.testing import (
import pytest
def test_poly1d_resolution(self):
    p = np.poly1d([1.0, 2, 3])
    q = np.poly1d([3.0, 2, 1])
    assert_equal(p(0), 3.0)
    assert_equal(p(5), 38.0)
    assert_equal(q(0), 1.0)
    assert_equal(q(5), 86.0)