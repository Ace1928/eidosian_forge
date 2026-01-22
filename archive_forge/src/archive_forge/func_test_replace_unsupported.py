from itertools import product
from itertools import permutations
from numba import njit, typeof
from numba.core import types
import unittest
from numba.tests.support import (TestCase, no_pyobj_flags, MemoryLeakMixin)
from numba.core.errors import TypingError, UnsupportedError
from numba.cpython.unicode import _MAX_UNICODE
from numba.core.types.functions import _header_lead
from numba.extending import overload
def test_replace_unsupported(self):

    def pyfunc(s, x, y, count):
        return s.replace(x, y, count)
    cfunc = njit(pyfunc)
    with self.assertRaises(TypingError) as raises:
        cfunc('ababababab', 'ba', 'qqq', 3.5)
    msg = 'Unsupported parameters. The parameters must be Integer.'
    self.assertIn(msg, str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc('ababababab', 0, 'qqq', 3)
    msg = 'The object must be a UnicodeType.'
    self.assertIn(msg, str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc('ababababab', 'ba', 0, 3)
    msg = 'The object must be a UnicodeType.'
    self.assertIn(msg, str(raises.exception))