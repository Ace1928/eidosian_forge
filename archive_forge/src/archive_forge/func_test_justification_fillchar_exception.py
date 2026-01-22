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
def test_justification_fillchar_exception(self):
    self.disable_leak_check()
    for pyfunc in [center_usecase_fillchar, ljust_usecase_fillchar, rjust_usecase_fillchar]:
        cfunc = njit(pyfunc)
        for fillchar in ['', '+0', 'quién', '处着']:
            with self.assertRaises(ValueError) as raises:
                cfunc(UNICODE_EXAMPLES[0], 20, fillchar)
            self.assertIn('The fill character must be exactly one', str(raises.exception))
        for fillchar in [1, 1.1]:
            with self.assertRaises(TypingError) as raises:
                cfunc(UNICODE_EXAMPLES[0], 20, fillchar)
            self.assertIn('The fillchar must be a UnicodeType', str(raises.exception))