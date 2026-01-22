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
def test_shape_invalid(self):
    max_int = np.iinfo(np.intc).max
    max_intp = np.iinfo(np.intp).max
    assert_raises(ValueError, np.dtype, [('a', 'f4', max_int // 4 + 1)])
    assert_raises(ValueError, np.dtype, [('a', 'f4', max_int + 1)])
    assert_raises(ValueError, np.dtype, [('a', 'f4', (max_int, 2))])
    assert_raises(ValueError, np.dtype, [('a', 'f4', max_intp + 1)])
    assert_raises(ValueError, np.dtype, [('a', 'f4', -1)])
    assert_raises(ValueError, np.dtype, [('a', 'f4', (-1, -1))])