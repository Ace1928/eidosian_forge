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
@pytest.mark.parametrize('dtype', ['Bool', 'Bytes0', 'Complex32', 'Complex64', 'Datetime64', 'Float16', 'Float32', 'Float64', 'Int8', 'Int16', 'Int32', 'Int64', 'Object0', 'Str0', 'Timedelta64', 'UInt8', 'UInt16', 'Uint32', 'UInt32', 'Uint64', 'UInt64', 'Void0', 'Float128', 'Complex128'])
def test_numeric_style_types_are_invalid(self, dtype):
    with assert_raises(TypeError):
        np.dtype(dtype)