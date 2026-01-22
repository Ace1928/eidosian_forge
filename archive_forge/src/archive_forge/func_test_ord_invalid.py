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
def test_ord_invalid(self):
    self.disable_leak_check()
    pyfunc = ord_usecase
    cfunc = njit(pyfunc)
    for func in (pyfunc, cfunc):
        for ch in ('', 'abc'):
            with self.assertRaises(TypeError) as raises:
                func(ch)
            self.assertIn('ord() expected a character', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(1.23)
    self.assertIn(_header_lead, str(raises.exception))