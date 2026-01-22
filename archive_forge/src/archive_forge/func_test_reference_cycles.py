import sys
import sysconfig
import weakref
from pathlib import Path
import pytest
import numpy as np
from numpy.ctypeslib import ndpointer, load_library, as_array
from numpy.testing import assert_, assert_array_equal, assert_raises, assert_equal
def test_reference_cycles(self):
    import ctypes
    N = 100
    a = np.arange(N, dtype=np.short)
    pnt = np.ctypeslib.as_ctypes(a)
    with np.testing.assert_no_gc_cycles():
        newpnt = ctypes.cast(pnt, ctypes.POINTER(ctypes.c_short))
        b = np.ctypeslib.as_array(newpnt, (N,))
        del newpnt, b