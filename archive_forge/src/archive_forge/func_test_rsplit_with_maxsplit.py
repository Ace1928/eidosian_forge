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
def test_rsplit_with_maxsplit(self):
    pyfuncs = [rsplit_with_maxsplit_usecase, rsplit_with_maxsplit_kwarg_usecase]
    CASES = [(' a ', None, 1), ('', '⚡', 1), ('abcabc', '⚡', 1), ('🐍⚡', '⚡', 1), ('🐍⚡🐍', '⚡', 1), ('abababa', 'a', 2), ('abababa', 'b', 1), ('abababa', 'c', 2), ('abababa', 'ab', 1), ('abababa', 'aba', 5)]
    messages = ['Results of "{}".rsplit("{}", {}) must be equal', 'Results of "{}".rsplit("{}", maxsplit={}) must be equal']
    for pyfunc, msg in zip(pyfuncs, messages):
        cfunc = njit(pyfunc)
        for test_str, sep, maxsplit in CASES:
            self.assertEqual(pyfunc(test_str, sep, maxsplit), cfunc(test_str, sep, maxsplit), msg=msg.format(test_str, sep, maxsplit))