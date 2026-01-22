import numpy as np
import unittest
from numba import jit, from_dtype
from numba.core import types
from numba.typed import Dict
from numba.tests.support import (TestCase, skip_ppc64le_issue4563)
def test_return_rstrip1(self):
    pyfunc = return_rstrip1
    cfunc = jit(nopython=True)(pyfunc)
    self._test(pyfunc, cfunc, np.array('  123  '), ())
    self._test(pyfunc, cfunc, np.array(b'  123  '), ())