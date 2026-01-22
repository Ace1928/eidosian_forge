import numpy as np
import unittest
from numba import jit, from_dtype
from numba.core import types
from numba.typed import Dict
from numba.tests.support import (TestCase, skip_ppc64le_issue4563)
def test_return_join(self):
    pyfunc = return_join
    cfunc = jit(nopython=True)(pyfunc)
    self._test(pyfunc, cfunc, np.array(','), (), np.array('abc'), (), np.array('123'), ())
    self._test(pyfunc, cfunc, np.array(','), (), np.array('abc'), (), ('123',), 0)
    self._test(pyfunc, cfunc, (',',), 0, np.array('abc'), (), np.array('123'), ())
    self._test(pyfunc, cfunc, (',',), 0, np.array('abc'), (), ('123',), 0)
    self._test(pyfunc, cfunc, np.array(b','), (), np.array(b'abc'), (), np.array(b'123'), ())
    self._test(pyfunc, cfunc, np.array(b','), (), np.array(b'abc'), (), (b'123',), 0)
    self._test(pyfunc, cfunc, (b',',), 0, np.array(b'abc'), (), np.array(b'123'), ())
    self._test(pyfunc, cfunc, (b',',), 0, np.array(b'abc'), (), (b'123',), 0)