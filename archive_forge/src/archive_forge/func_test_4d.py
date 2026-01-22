from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
def test_4d(self):
    s = readsav(path.join(DATA_PATH, 'array_float32_pointer_4d.sav'), verbose=False)
    assert_equal(s.array4d.shape, (4, 5, 8, 7))
    assert_(np.all(s.array4d == np.float32(4.0)))
    assert_(np.all(vect_id(s.array4d) == id(s.array4d[0, 0, 0, 0])))