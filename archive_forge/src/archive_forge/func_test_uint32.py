from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
def test_uint32(self):
    s = readsav(path.join(DATA_PATH, 'scalar_uint32.sav'), verbose=False)
    assert_identical(s.i32u, np.uint32(4294967233))