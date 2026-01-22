import copy
import sys
import gc
import tempfile
import pytest
from os import path
from io import BytesIO
from itertools import chain
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _no_tracing, requires_memory
from numpy.compat import asbytes, asunicode, pickle
def test_object_array_to_fixed_string(self):
    a = np.array(['abcdefgh', 'ijklmnop'], dtype=np.object_)
    b = np.array(a, dtype=(np.str_, 8))
    assert_equal(a, b)
    c = np.array(a, dtype=(np.str_, 5))
    assert_equal(c, np.array(['abcde', 'ijklm']))
    d = np.array(a, dtype=(np.str_, 12))
    assert_equal(a, d)
    e = np.empty((2,), dtype=(np.str_, 8))
    e[:] = a[:]
    assert_equal(a, e)