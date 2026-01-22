import numpy as np
import unittest
from numba import njit
from numba.core.errors import TypingError
from numba import jit, typeof
from numba.core import types
from numba.tests.support import TestCase
def test_issue_1850(self):
    """
        This issue is caused by an unresolved bug in numpy since version 1.6.
        See numpy GH issue #3147.
        """
    constarr = np.array([86])

    def pyfunc():
        return constarr[0]
    cfunc = njit(())(pyfunc)
    out = cfunc()
    self.assertEqual(out, 86)