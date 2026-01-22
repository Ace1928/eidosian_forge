from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
def test_6d(self):
    s = readsav(path.join(DATA_PATH, 'array_float32_pointer_6d.sav'), verbose=False)
    assert_equal(s.array6d.shape, (3, 6, 4, 5, 3, 4))
    assert_(np.all(s.array6d == np.float32(4.0)))
    assert_(np.all(vect_id(s.array6d) == id(s.array6d[0, 0, 0, 0, 0, 0])))