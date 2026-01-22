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
def test_endswith_with_start_end(self):
    pyfunc = endswith_with_start_end_usecase
    cfunc = njit(pyfunc)
    cpython_str = ['hello', 'helloworld', '']
    cpython_subs = ['he', 'hello', 'helloworld', 'ello', '', 'lowo', 'lo', 'he', 'lo', 'o']
    extra_subs = ['hellohellohello', ' ']
    for s in cpython_str + UNICODE_EXAMPLES:
        default_subs = ['', 'x', s[:-2], s[3:], s, s + s]
        for sub_str in cpython_subs + default_subs + extra_subs:
            for start in list(range(-20, 20)) + [None]:
                for end in list(range(-20, 20)) + [None]:
                    msg = 'Results "{}".endswith("{}", {}, {})                               must be equal'
                    self.assertEqual(pyfunc(s, sub_str, start, end), cfunc(s, sub_str, start, end), msg=msg.format(s, sub_str, start, end))