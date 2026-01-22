from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
def test_scalars(self):
    s = readsav(path.join(DATA_PATH, 'struct_pointers.sav'), verbose=False)
    assert_identical(s.pointers.g, np.array(np.float32(4.0), dtype=np.object_))
    assert_identical(s.pointers.h, np.array(np.float32(4.0), dtype=np.object_))
    assert_(id(s.pointers.g[0]) == id(s.pointers.h[0]))