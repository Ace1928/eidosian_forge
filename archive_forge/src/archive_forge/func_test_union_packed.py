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
def test_union_packed(self):

    class Struct(ctypes.Structure):
        _fields_ = [('one', ctypes.c_uint8), ('two', ctypes.c_uint32)]
        _pack_ = 1

    class Union(ctypes.Union):
        _pack_ = 1
        _fields_ = [('a', ctypes.c_uint8), ('b', ctypes.c_uint16), ('c', ctypes.c_uint32), ('d', Struct)]
    expected = np.dtype(dict(names=['a', 'b', 'c', 'd'], formats=['u1', np.uint16, np.uint32, [('one', 'u1'), ('two', np.uint32)]], offsets=[0, 0, 0, 0], itemsize=ctypes.sizeof(Union)))
    self.check(Union, expected)