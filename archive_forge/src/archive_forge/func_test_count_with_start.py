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
def test_count_with_start(self):
    pyfunc = count_with_start_usecase
    cfunc = njit(pyfunc)
    error_msg = '%s\n%s' % ("'{0}'.py_count('{1}', {2}) = {3}", "'{0}'.c_count('{1}', {2}) = {4}")
    for s, sub in UNICODE_COUNT_EXAMPLES:
        for i in range(-18, 18):
            py_result = pyfunc(s, sub, i)
            c_result = cfunc(s, sub, i)
            self.assertEqual(py_result, c_result, error_msg.format(s, sub, i, py_result, c_result))
        py_result = pyfunc(s, sub, None)
        c_result = cfunc(s, sub, None)
        self.assertEqual(py_result, c_result, error_msg.format(s, sub, None, py_result, c_result))