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
def test_rfind_with_start_end(self):
    pyfunc = rfind_with_start_end_usecase
    cfunc = njit(pyfunc)
    starts = list(range(-20, 20)) + [None]
    ends = list(range(-20, 20)) + [None]
    for s in UNICODE_EXAMPLES:
        for sub_str in ['', 'xx', s[:-2], s[3:], s]:
            for start, end in product(starts, ends):
                msg = 'Results of "{}".rfind("{}", {}, {}) must be equal'
                self.assertEqual(pyfunc(s, sub_str, start, end), cfunc(s, sub_str, start, end), msg=msg.format(s, sub_str, start, end))