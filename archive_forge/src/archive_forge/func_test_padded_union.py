import sys
import sysconfig
import weakref
from pathlib import Path
import pytest
import numpy as np
from numpy.ctypeslib import ndpointer, load_library, as_array
from numpy.testing import assert_, assert_array_equal, assert_raises, assert_equal
def test_padded_union(self):
    dt = np.dtype(dict(names=['a', 'b'], offsets=[0, 0], formats=[np.uint16, np.uint32], itemsize=5))
    ct = np.ctypeslib.as_ctypes_type(dt)
    assert_(issubclass(ct, ctypes.Union))
    assert_equal(ctypes.sizeof(ct), dt.itemsize)
    assert_equal(ct._fields_, [('a', ctypes.c_uint16), ('b', ctypes.c_uint32), ('', ctypes.c_char * 5)])