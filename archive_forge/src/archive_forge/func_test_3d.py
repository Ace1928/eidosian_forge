from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
def test_3d(self):
    s = readsav(path.join(DATA_PATH, 'array_float32_pointer_3d.sav'), verbose=False)
    assert_equal(s.array3d.shape, (11, 22, 12))
    assert_(np.all(s.array3d == np.float32(4.0)))
    assert_(np.all(vect_id(s.array3d) == id(s.array3d[0, 0, 0])))