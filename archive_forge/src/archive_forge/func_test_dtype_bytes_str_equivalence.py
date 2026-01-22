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
@pytest.mark.parametrize('value', ['m8', 'M8', 'datetime64', 'timedelta64', 'i4, (2,3)f8, f4', 'a3, 3u8, (3,4)a10', '>f', '<f', '=f', '|f'])
def test_dtype_bytes_str_equivalence(self, value):
    bytes_value = value.encode('ascii')
    from_bytes = np.dtype(bytes_value)
    from_str = np.dtype(value)
    assert_dtype_equal(from_bytes, from_str)