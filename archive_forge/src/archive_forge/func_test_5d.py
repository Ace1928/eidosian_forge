from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
def test_5d(self):
    s = readsav(path.join(DATA_PATH, 'array_float32_pointer_5d.sav'), verbose=False)
    assert_equal(s.array5d.shape, (4, 3, 4, 6, 5))
    assert_(np.all(s.array5d == np.float32(4.0)))
    assert_(np.all(vect_id(s.array5d) == id(s.array5d[0, 0, 0, 0, 0])))