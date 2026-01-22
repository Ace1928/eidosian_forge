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
def test_structured_object_item_setting(self, dt, pat, count, singleton):
    """Structured object reference counting for simple item setting"""
    one = 1
    gc.collect()
    before = sys.getrefcount(singleton)
    arr = np.array([pat] * 3, dt)
    assert sys.getrefcount(singleton) - before == count * 3
    before2 = sys.getrefcount(one)
    arr[...] = one
    after2 = sys.getrefcount(one)
    assert after2 - before2 == count * 3
    del arr
    gc.collect()
    assert sys.getrefcount(one) == before2
    assert sys.getrefcount(singleton) == before