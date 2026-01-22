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
def test_startswith_exception_invalid_args(self):
    msg_invalid_prefix = "The arg 'prefix' should be a string or a tuple of strings"
    with self.assertRaisesRegex(TypingError, msg_invalid_prefix):
        cfunc = njit(startswith_usecase)
        cfunc('hello', (1, 'he'))
    msg_invalid_start = "When specified, the arg 'start' must be an Integer or None"
    with self.assertRaisesRegex(TypingError, msg_invalid_start):
        cfunc = njit(startswith_with_start_only_usecase)
        cfunc('hello', 'he', 'invalid start')
    msg_invalid_end = "When specified, the arg 'end' must be an Integer or None"
    with self.assertRaisesRegex(TypingError, msg_invalid_end):
        cfunc = njit(startswith_with_start_end_usecase)
        cfunc('hello', 'he', 0, 'invalid end')