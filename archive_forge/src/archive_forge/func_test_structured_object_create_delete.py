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
@pytest.mark.parametrize(['creation_func', 'creation_obj'], [pytest.param(np.empty, None, marks=pytest.mark.skip("unreliable due to python's behaviour")), (np.ones, 1), (np.zeros, 0)])
def test_structured_object_create_delete(self, dt, pat, count, singleton, creation_func, creation_obj):
    """Structured object reference counting in creation and deletion"""
    gc.collect()
    before = sys.getrefcount(creation_obj)
    arr = creation_func(3, dt)
    now = sys.getrefcount(creation_obj)
    assert now - before == count * 3
    del arr
    now = sys.getrefcount(creation_obj)
    assert now == before