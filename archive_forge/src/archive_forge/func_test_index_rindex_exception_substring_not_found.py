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
def test_index_rindex_exception_substring_not_found(self):
    self.disable_leak_check()
    unicode_examples = ['ascii', '12345', '1234567890', '¡Y tú quién te crees?', '大处着眼，小处着手。']
    pyfuncs = [index_with_start_end_usecase, rindex_with_start_end_usecase]
    for pyfunc in pyfuncs:
        cfunc = njit(pyfunc)
        for s in unicode_examples:
            l = len(s)
            cases = [('', list(range(l + 1, 10)), [l]), (s[:-2], [0], list(range(l - 2))), (s[3:], list(range(4, 10)), [l]), (s, [None], list(range(l)))]
            for sub_str, starts, ends in cases:
                for start, end in product(starts, ends):
                    for func in [pyfunc, cfunc]:
                        with self.assertRaises(ValueError) as raises:
                            func(s, sub_str, start, end)
                        msg = 'substring not found'
                        self.assertIn(msg, str(raises.exception))