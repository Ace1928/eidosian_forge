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
def test_index_rindex_with_start_only(self):
    pyfuncs = [index_with_start_only_usecase, rindex_with_start_only_usecase]
    messages = ['Results "{}".index("{}", {}) must be equal', 'Results "{}".rindex("{}", {}) must be equal']
    unicode_examples = ['ascii', '12345', '1234567890', '¡Y tú quién te crees?', '大处着眼，小处着手。']
    for pyfunc, msg in zip(pyfuncs, messages):
        cfunc = njit(pyfunc)
        for s in unicode_examples:
            l = len(s)
            cases = [('', list(range(-10, l + 1))), (s[:-2], [0] + list(range(-10, 1 - l))), (s[3:], list(range(4)) + list(range(-10, 4 - l))), (s, [0] + list(range(-10, 1 - l)))]
            for sub_str, starts in cases:
                for start in starts + [None]:
                    self.assertEqual(pyfunc(s, sub_str, start), cfunc(s, sub_str, start), msg=msg.format(s, sub_str, start))