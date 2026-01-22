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
@pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
def test_take_refcount(self):
    a = np.arange(16, dtype=float)
    a.shape = (4, 4)
    lut = np.ones((5 + 3, 4), float)
    rgba = np.empty(shape=a.shape + (4,), dtype=lut.dtype)
    c1 = sys.getrefcount(rgba)
    try:
        lut.take(a, axis=0, mode='clip', out=rgba)
    except TypeError:
        pass
    c2 = sys.getrefcount(rgba)
    assert_equal(c1, c2)