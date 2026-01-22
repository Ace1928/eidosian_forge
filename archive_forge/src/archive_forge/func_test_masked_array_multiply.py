import numpy as np
from numpy.testing import (
def test_masked_array_multiply(self):
    a = np.ma.zeros((4, 1))
    a[2, 0] = np.ma.masked
    b = np.zeros((4, 2))
    a * b
    b * a