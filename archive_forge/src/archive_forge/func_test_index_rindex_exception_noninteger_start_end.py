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
def test_index_rindex_exception_noninteger_start_end(self):
    accepted = (types.Integer, types.NoneType)
    pyfuncs = [index_with_start_end_usecase, rindex_with_start_end_usecase]
    for pyfunc in pyfuncs:
        cfunc = njit(pyfunc)
        for start, end, name in [(0.1, 5, 'start'), (0, 0.5, 'end')]:
            with self.assertRaises(TypingError) as raises:
                cfunc('ascii', 'sci', start, end)
            msg = '"{}" must be {}, not float'.format(name, accepted)
            self.assertIn(msg, str(raises.exception))