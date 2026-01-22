from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
def test_int64(self):
    s = readsav(path.join(DATA_PATH, 'scalar_int64.sav'), verbose=False)
    assert_identical(s.i64s, np.int64(-9223372036854774567))