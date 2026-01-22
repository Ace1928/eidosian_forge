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
def test_replace_with_count(self):
    pyfunc = replace_with_count_usecase
    cfunc = njit(pyfunc)
    CASES = [('abc', '', 'A'), ('', '⚡', 'A'), ('abcabc', '⚡', 'A'), ('🐍⚡', '⚡', 'A'), ('🐍⚡🐍', '⚡', 'A'), ('abababa', 'a', 'A'), ('abababa', 'b', 'A'), ('abababa', 'c', 'A'), ('abababa', 'ab', 'A'), ('abababa', 'aba', 'A')]
    count_test = [-1, 1, 0, 5]
    for test_str, old_str, new_str in CASES:
        for count in count_test:
            self.assertEqual(pyfunc(test_str, old_str, new_str, count), cfunc(test_str, old_str, new_str, count), "'%s'.replace('%s', '%s', '%s')?" % (test_str, old_str, new_str, count))