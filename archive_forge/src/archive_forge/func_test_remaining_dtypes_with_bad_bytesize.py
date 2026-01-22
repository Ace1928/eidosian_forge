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
def test_remaining_dtypes_with_bad_bytesize(self):
    assert np.dtype('int0') is np.dtype('intp')
    assert np.dtype('uint0') is np.dtype('uintp')
    assert np.dtype('bool8') is np.dtype('bool')
    assert np.dtype('bytes0') is np.dtype('bytes')
    assert np.dtype('str0') is np.dtype('str')
    assert np.dtype('object0') is np.dtype('object')