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
def test_isalnum(self):

    def pyfunc(x):
        return x.isalnum()
    cfunc = njit(pyfunc)
    cpython = ['𐐁', '𐐧', '𐐩', '𐑎', '𝟶', '𑁦', '𐒠', '🄇']
    extras = ['\ud800', '\udfff', '\ud800\ud800', '\udfff\udfff', 'a\ud800b\udfff', 'a\udfffb\ud800', 'a\ud800b\udfffa', 'a\udfffb\ud800a']
    msg = 'Results of "{}".isalnum() must be equal'
    for s in UNICODE_EXAMPLES + [''] + extras + cpython:
        self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))