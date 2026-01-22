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
def test_partition_exception_invalid_sep(self):
    self.disable_leak_check()
    pyfunc = partition_usecase
    cfunc = njit(pyfunc)
    for func in [pyfunc, cfunc]:
        with self.assertRaises(ValueError) as raises:
            func('a', '')
        self.assertIn('empty separator', str(raises.exception))
    accepted_types = (types.UnicodeType, types.UnicodeCharSeq)
    with self.assertRaises(TypingError) as raises:
        cfunc('a', None)
    msg = '"sep" must be {}, not none'.format(accepted_types)
    self.assertIn(msg, str(raises.exception))