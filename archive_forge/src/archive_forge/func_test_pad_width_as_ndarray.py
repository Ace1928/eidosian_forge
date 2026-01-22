import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
def test_pad_width_as_ndarray(self):
    a = np.arange(12)
    a = np.reshape(a, (4, 3))
    a = np.pad(a, np.array(((2, 3), (3, 2))), 'edge')
    b = np.array([[0, 0, 0, 0, 1, 2, 2, 2], [0, 0, 0, 0, 1, 2, 2, 2], [0, 0, 0, 0, 1, 2, 2, 2], [3, 3, 3, 3, 4, 5, 5, 5], [6, 6, 6, 6, 7, 8, 8, 8], [9, 9, 9, 9, 10, 11, 11, 11], [9, 9, 9, 9, 10, 11, 11, 11], [9, 9, 9, 9, 10, 11, 11, 11], [9, 9, 9, 9, 10, 11, 11, 11]])
    assert_array_equal(a, b)