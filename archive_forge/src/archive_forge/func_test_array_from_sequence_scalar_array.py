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
def test_array_from_sequence_scalar_array(self):
    a = np.array((np.ones(2), np.array(2)), dtype=object)
    assert_equal(a.shape, (2,))
    assert_equal(a.dtype, np.dtype(object))
    assert_equal(a[0], np.ones(2))
    assert_equal(a[1], np.array(2))
    a = np.array(((1,), np.array(1)), dtype=object)
    assert_equal(a.shape, (2,))
    assert_equal(a.dtype, np.dtype(object))
    assert_equal(a[0], (1,))
    assert_equal(a[1], np.array(1))