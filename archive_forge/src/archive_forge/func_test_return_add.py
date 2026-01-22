import numpy as np
import unittest
from numba import jit, from_dtype
from numba.core import types
from numba.typed import Dict
from numba.tests.support import (TestCase, skip_ppc64le_issue4563)
def test_return_add(self):
    pyfunc = return_add
    cfunc = jit(nopython=True)(pyfunc)
    self._test(pyfunc, cfunc, np.array('ab'), (), np.array('cd'), ())
    self._test(pyfunc, cfunc, np.array('ab'), (), ('cd',), 0)
    self._test(pyfunc, cfunc, ('ab',), 0, np.array('cd'), ())
    self._test(pyfunc, cfunc, np.array(b'ab'), (), np.array(b'cd'), ())
    self._test(pyfunc, cfunc, np.array(b'ab'), (), (b'cd',), 0)
    self._test(pyfunc, cfunc, (b'ab',), 0, np.array(b'cd'), ())