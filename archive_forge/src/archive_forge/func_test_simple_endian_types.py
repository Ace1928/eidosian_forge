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
def test_simple_endian_types(self):
    self.check(ctypes.c_uint16.__ctype_le__, np.dtype('<u2'))
    self.check(ctypes.c_uint16.__ctype_be__, np.dtype('>u2'))
    self.check(ctypes.c_uint8.__ctype_le__, np.dtype('u1'))
    self.check(ctypes.c_uint8.__ctype_be__, np.dtype('u1'))