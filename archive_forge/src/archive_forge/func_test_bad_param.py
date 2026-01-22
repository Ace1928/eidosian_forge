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
def test_bad_param(self):
    assert_raises(ValueError, np.dtype, {'names': ['f0', 'f1'], 'formats': ['i4', 'i1'], 'offsets': [0, 4], 'itemsize': 4})
    assert_raises(ValueError, np.dtype, {'names': ['f0', 'f1'], 'formats': ['i4', 'i1'], 'offsets': [0, 4], 'itemsize': 9}, align=True)
    assert_raises(ValueError, np.dtype, {'names': ['f0', 'f1'], 'formats': ['i1', 'f4'], 'offsets': [0, 2]}, align=True)