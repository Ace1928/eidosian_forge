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
@pytest.mark.parametrize('align_flag', [False, True])
def test_multifield_index(self, align_flag):
    dt = np.dtype([(('title', 'col1'), '<U20'), ('A', '<f8'), ('B', '<f8')], align=align_flag)
    dt_sub = dt[['B', 'col1']]
    assert_equal(dt_sub, np.dtype({'names': ['B', 'col1'], 'formats': ['<f8', '<U20'], 'offsets': [88, 0], 'titles': [None, 'title'], 'itemsize': 96}))
    assert_equal(dt_sub.isalignedstruct, align_flag)
    dt_sub = dt[['B']]
    assert_equal(dt_sub, np.dtype({'names': ['B'], 'formats': ['<f8'], 'offsets': [88], 'itemsize': 96}))
    assert_equal(dt_sub.isalignedstruct, align_flag)
    dt_sub = dt[[]]
    assert_equal(dt_sub, np.dtype({'names': [], 'formats': [], 'offsets': [], 'itemsize': 96}))
    assert_equal(dt_sub.isalignedstruct, align_flag)
    assert_raises(TypeError, operator.getitem, dt, ())
    assert_raises(TypeError, operator.getitem, dt, [1, 2, 3])
    assert_raises(TypeError, operator.getitem, dt, ['col1', 2])
    assert_raises(KeyError, operator.getitem, dt, ['fake'])
    assert_raises(KeyError, operator.getitem, dt, ['title'])
    assert_raises(ValueError, operator.getitem, dt, ['col1', 'col1'])