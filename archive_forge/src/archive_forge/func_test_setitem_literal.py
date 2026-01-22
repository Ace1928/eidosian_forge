import numpy as np
import unittest
from numba import jit, from_dtype
from numba.core import types
from numba.typed import Dict
from numba.tests.support import (TestCase, skip_ppc64le_issue4563)
def test_setitem_literal(self):
    pyfunc = setitem_literal
    cfunc = jit(nopython=True)(pyfunc)
    x1 = np.array('ABC')
    x2 = np.array('ABC')
    y1 = pyfunc(x1, ())
    y2 = cfunc(x2, ())
    self.assertPreciseEqual(x1, x2)
    self.assertPreciseEqual(y1, y2)
    x1 = np.array(['ABC', '5678'])
    x2 = np.array(['ABC', '5678'])
    y1 = pyfunc(x1, 0)
    y2 = cfunc(x2, 0)
    self.assertPreciseEqual(x1, x2)
    self.assertPreciseEqual(y1, y2)
    x1 = np.array(['ABC', '5678'])
    x2 = np.array(['ABC', '5678'])
    y1 = pyfunc(x1, 1)
    y2 = cfunc(x2, 1)
    self.assertPreciseEqual(x1, x2)
    self.assertPreciseEqual(y1, y2)