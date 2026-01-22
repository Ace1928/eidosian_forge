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
def test_sparse_field_assignment(self):
    arr = np.zeros(3, self.dtype)
    sparse_arr = arr.view(self.sparse_dtype)
    sparse_arr[...] = np.finfo(np.float32).max
    assert_array_equal(arr['a']['aa'], np.zeros((3, 2, 3)))