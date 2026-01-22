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
def test_in(self, flags=no_pyobj_flags):
    pyfunc = in_usecase
    cfunc = njit(pyfunc)
    for a in UNICODE_EXAMPLES:
        extras = ['', 'xx', a[::-1], a[:-2], a[3:], a, a + a]
        for substr in extras:
            self.assertEqual(pyfunc(substr, a), cfunc(substr, a), "'%s' in '%s'?" % (substr, a))