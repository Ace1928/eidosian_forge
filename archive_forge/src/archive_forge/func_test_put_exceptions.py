import numpy as np
from numpy.testing import (
def test_put_exceptions(self):
    a = np.zeros((5, 5))
    assert_raises(IndexError, a.put, 100, 0)
    a = np.zeros((5, 5), dtype=object)
    assert_raises(IndexError, a.put, 100, 0)
    a = np.zeros((5, 5, 0))
    assert_raises(IndexError, a.put, 100, 0)
    a = np.zeros((5, 5, 0), dtype=object)
    assert_raises(IndexError, a.put, 100, 0)