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
def test_repeat(self, flags=no_pyobj_flags):
    pyfunc = repeat_usecase
    cfunc = njit(pyfunc)
    for a in UNICODE_EXAMPLES:
        for b in (-1, 0, 1, 2, 3, 4, 5, 7, 8, 15, 70):
            self.assertEqual(pyfunc(a, b), cfunc(a, b))
            self.assertEqual(pyfunc(b, a), cfunc(b, a))