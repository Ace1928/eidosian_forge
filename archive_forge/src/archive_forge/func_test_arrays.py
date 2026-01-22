from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
def test_arrays(self):
    s = readsav(path.join(DATA_PATH, 'struct_pointer_arrays.sav'), verbose=False)
    assert_array_identical(s.arrays.g[0], np.repeat(np.float32(4.0), 2).astype(np.object_))
    assert_array_identical(s.arrays.h[0], np.repeat(np.float32(4.0), 3).astype(np.object_))
    assert_(np.all(vect_id(s.arrays.g[0]) == id(s.arrays.g[0][0])))
    assert_(np.all(vect_id(s.arrays.h[0]) == id(s.arrays.h[0][0])))
    assert_(id(s.arrays.g[0][0]) == id(s.arrays.h[0][0]))