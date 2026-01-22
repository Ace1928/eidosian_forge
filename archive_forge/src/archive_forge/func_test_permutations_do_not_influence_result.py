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
@pytest.mark.parametrize(['dtypes', 'expected'], [([np.uint16, np.int16, np.float16], np.float32), ([np.uint16, np.int8, np.float16], np.float32), ([np.uint8, np.int16, np.float16], np.float32), ([1, 1, np.float64], np.float64), ([1, 1.0, np.complex128], np.complex128), ([1, 1j, np.float64], np.complex128), ([1.0, 1.0, np.int64], np.float64), ([1.0, 1j, np.float64], np.complex128), ([1j, 1j, np.float64], np.complex128), ([1, True, np.bool_], np.int_)])
def test_permutations_do_not_influence_result(self, dtypes, expected):
    for perm in permutations(dtypes):
        assert np.result_type(*perm) == expected