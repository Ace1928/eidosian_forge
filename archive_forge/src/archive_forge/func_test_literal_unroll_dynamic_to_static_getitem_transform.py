import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_literal_unroll_dynamic_to_static_getitem_transform(self):
    keys = ('a', 'b', 'c')
    n = 5

    def pyfunc(rec):
        x = np.zeros((n,))
        for o in literal_unroll(keys):
            x += rec[o]
        return x
    dt = np.float64
    ldd = [np.arange(dt(n)) for x in keys]
    ldk = [(x, np.float64) for x in keys]
    rec = np.rec.fromarrays(ldd, dtype=ldk)
    expected = pyfunc(rec)
    got = njit(pyfunc)(rec)
    np.testing.assert_allclose(expected, got)