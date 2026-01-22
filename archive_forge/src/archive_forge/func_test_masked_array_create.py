import numpy as np
from numpy.testing import (
def test_masked_array_create(self):
    x = np.ma.masked_array([0, 1, 2, 3, 0, 4, 5, 6], mask=[0, 0, 0, 1, 1, 1, 0, 0])
    assert_array_equal(np.ma.nonzero(x), [[1, 2, 6, 7]])