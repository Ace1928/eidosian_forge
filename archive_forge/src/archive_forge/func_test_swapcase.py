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
def test_swapcase(self):

    def pyfunc(x):
        return x.swapcase()
    cfunc = njit(pyfunc)
    cpython = ['𐑏', '𐐧', '𐑏𐑏', '𐐧𐑏', '𐑏𐐧', 'X𐐧x𐑏', 'ﬁ', 'İ', 'Σ', 'ͅΣ', 'AͅΣ', 'AͅΣa', 'AͅΣ', 'AΣͅ', 'Σͅ ', 'Σ', 'ß', 'ῒ']
    cpython_extras = ['𐀀\U00100000']
    msg = 'Results of "{}".swapcase() must be equal'
    for s in UNICODE_EXAMPLES + [''] + cpython + cpython_extras:
        self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))