import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, IS_PYPY
def test_size1dims_arrays(self):
    x = np.ndarray(dtype='f8', shape=(10, 5, 1), strides=(8, 80, 4), buffer=np.ones(1000, dtype=np.uint8), order='F')
    y = np.from_dlpack(x)
    assert_array_equal(x, y)