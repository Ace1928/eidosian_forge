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
def test_getitem_scalar_kind(self):
    pyfunc = getitem_check_kind_usecase
    cfunc = njit(pyfunc)
    samples = ['aሴ', '¡着']
    for s in samples:
        for i in range(-len(s), len(s)):
            self.assertEqual(pyfunc(s, i), cfunc(s, i), "'%s'[%d]?" % (s, i))