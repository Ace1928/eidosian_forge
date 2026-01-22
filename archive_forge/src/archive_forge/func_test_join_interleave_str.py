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
def test_join_interleave_str(self):
    pyfunc = join_usecase
    cfunc = njit(pyfunc)
    CASES = [('abc', '123'), ('🐍🐍🐍', '⚡⚡')]
    for sep, parts in CASES:
        self.assertEqual(pyfunc(sep, parts), cfunc(sep, parts), "'%s'.join('%s')?" % (sep, parts))