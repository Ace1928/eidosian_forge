import sys
import operator
import pytest
import ctypes
import gc
import types
from typing import Any
import numpy as np
import numpy.dtypes
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import create_custom_field_dtype
from numpy.testing import (
from numpy.compat import pickle
from itertools import permutations
import random
import hypothesis
from hypothesis.extra import numpy as hynp
def test_fields_by_index(self):
    dt = np.dtype([('a', np.int8), ('b', np.float32, 3)])
    assert_dtype_equal(dt[0], np.dtype(np.int8))
    assert_dtype_equal(dt[1], np.dtype((np.float32, 3)))
    assert_dtype_equal(dt[-1], dt[1])
    assert_dtype_equal(dt[-2], dt[0])
    assert_raises(IndexError, lambda: dt[-3])
    assert_raises(TypeError, operator.getitem, dt, 3.0)
    assert_equal(dt[1], dt[np.int8(1)])