from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
def test_pointers_replicated(self):
    s = readsav(path.join(DATA_PATH, 'struct_pointers_replicated.sav'), verbose=False)
    assert_identical(s.pointers_rep.g, np.repeat(np.float32(4.0), 5).astype(np.object_))
    assert_identical(s.pointers_rep.h, np.repeat(np.float32(4.0), 5).astype(np.object_))
    assert_(np.all(vect_id(s.pointers_rep.g) == vect_id(s.pointers_rep.h)))