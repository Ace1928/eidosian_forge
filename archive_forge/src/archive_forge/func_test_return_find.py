import numpy as np
import unittest
from numba import jit, from_dtype
from numba.core import types
from numba.typed import Dict
from numba.tests.support import (TestCase, skip_ppc64le_issue4563)
def test_return_find(self):
    pyfunc = return_find
    cfunc = jit(nopython=True)(pyfunc)
    self._test(pyfunc, cfunc, np.array('1234'), (), np.array('23'), ())
    self._test(pyfunc, cfunc, np.array('1234'), (), ('23',), 0)
    self._test(pyfunc, cfunc, ('1234',), 0, np.array('23'), ())
    self._test(pyfunc, cfunc, np.array(b'1234'), (), np.array(b'23'), ())
    self._test(pyfunc, cfunc, np.array(b'1234'), (), (b'23',), 0)
    self._test(pyfunc, cfunc, (b'1234',), 0, np.array(b'23'), ())