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
@pytest.mark.parametrize(['dt', 'pat', 'count', 'singleton'], iter_struct_object_dtypes())
def test_structured_object_take_and_repeat(self, dt, pat, count, singleton):
    """Structured object reference counting for specialized functions.
        The older functions such as take and repeat use different code paths
        then item setting (when writing this).
        """
    indices = [0, 1]
    arr = np.array([pat] * 3, dt)
    gc.collect()
    before = sys.getrefcount(singleton)
    res = arr.take(indices)
    after = sys.getrefcount(singleton)
    assert after - before == count * 2
    new = res.repeat(10)
    gc.collect()
    after_repeat = sys.getrefcount(singleton)
    assert after_repeat - after == count * 2 * 10