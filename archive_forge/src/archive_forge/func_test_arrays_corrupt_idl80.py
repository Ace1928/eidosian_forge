from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
def test_arrays_corrupt_idl80(self):
    with suppress_warnings() as sup:
        sup.filter(UserWarning, 'Not able to verify number of bytes from header')
        s = readsav(path.join(DATA_PATH, 'struct_arrays_byte_idl80.sav'), verbose=False)
    assert_identical(s.y.x[0], np.array([55, 66], dtype=np.uint8))