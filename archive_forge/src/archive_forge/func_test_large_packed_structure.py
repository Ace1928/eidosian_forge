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
def test_large_packed_structure(self):

    class PackedStructure(ctypes.Structure):
        _pack_ = 2
        _fields_ = [('a', ctypes.c_uint8), ('b', ctypes.c_uint16), ('c', ctypes.c_uint8), ('d', ctypes.c_uint16), ('e', ctypes.c_uint32), ('f', ctypes.c_uint32), ('g', ctypes.c_uint8)]
    expected = np.dtype(dict(formats=[np.uint8, np.uint16, np.uint8, np.uint16, np.uint32, np.uint32, np.uint8], offsets=[0, 2, 4, 6, 8, 12, 16], names=['a', 'b', 'c', 'd', 'e', 'f', 'g'], itemsize=18))
    self.check(PackedStructure, expected)