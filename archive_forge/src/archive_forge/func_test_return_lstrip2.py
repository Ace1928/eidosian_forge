import numpy as np
import unittest
from numba import jit, from_dtype
from numba.core import types
from numba.typed import Dict
from numba.tests.support import (TestCase, skip_ppc64le_issue4563)
def test_return_lstrip2(self):
    pyfunc = return_lstrip2
    cfunc = jit(nopython=True)(pyfunc)
    self._test(pyfunc, cfunc, np.array('  123  '), (), np.array(' '), ())
    self._test(pyfunc, cfunc, np.array('  123  '), (), (' ',), 0)
    self._test(pyfunc, cfunc, ('  123  ',), 0, np.array(' '), ())
    self._test(pyfunc, cfunc, np.array(b'  123  '), (), np.array(b' '), ())
    self._test(pyfunc, cfunc, np.array(b'  123  '), (), (b' ',), 0)
    self._test(pyfunc, cfunc, (b'  123  ',), 0, np.array(b' '), ())