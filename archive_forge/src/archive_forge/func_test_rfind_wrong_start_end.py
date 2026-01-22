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
def test_rfind_wrong_start_end(self):
    cfunc = njit(rfind_with_start_end_usecase)
    accepted_types = (types.Integer, types.NoneType)
    for s in UNICODE_EXAMPLES:
        for sub_str in ['', 'xx', s[:-2], s[3:], s]:
            for start, end in product([0.1, False], [-1, 1]):
                with self.assertRaises(TypingError) as raises:
                    cfunc(s, sub_str, start, end)
                msg = '"start" must be {}'.format(accepted_types)
                self.assertIn(msg, str(raises.exception))
            for start, end in product([-1, 1], [-0.1, True]):
                with self.assertRaises(TypingError) as raises:
                    cfunc(s, sub_str, start, end)
                msg = '"end" must be {}'.format(accepted_types)
                self.assertIn(msg, str(raises.exception))