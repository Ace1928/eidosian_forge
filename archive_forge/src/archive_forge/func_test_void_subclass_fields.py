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
def test_void_subclass_fields(self):
    dt = np.dtype((np.record, [('a', '<u2')]))
    assert_equal(repr(dt), "dtype((numpy.record, [('a', '<u2')]))")
    assert_equal(str(dt), "(numpy.record, [('a', '<u2')])")
    assert_equal(dt.name, 'record16')