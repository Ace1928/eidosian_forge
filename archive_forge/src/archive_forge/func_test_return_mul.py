import numpy as np
import unittest
from numba import jit, from_dtype
from numba.core import types
from numba.typed import Dict
from numba.tests.support import (TestCase, skip_ppc64le_issue4563)
def test_return_mul(self):
    pyfunc = return_mul
    cfunc = jit(nopython=True)(pyfunc)
    self._test(pyfunc, cfunc, np.array('ab'), (), (5,), 0)
    self._test(pyfunc, cfunc, (5,), 0, np.array('ab'), ())
    self._test(pyfunc, cfunc, np.array(b'ab'), (), (5,), 0)
    self._test(pyfunc, cfunc, (5,), 0, np.array(b'ab'), ())