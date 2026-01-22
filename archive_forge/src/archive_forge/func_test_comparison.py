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
def test_comparison(self):

    def pyfunc(option, x, y):
        if option == '==':
            return x == y
        elif option == '!=':
            return x != y
        elif option == '<':
            return x < y
        elif option == '>':
            return x > y
        elif option == '<=':
            return x <= y
        elif option == '>=':
            return x >= y
        else:
            return None
    cfunc = njit(pyfunc)
    for x, y in permutations(UNICODE_ORDERING_EXAMPLES, r=2):
        for cmpop in ['==', '!=', '<', '>', '<=', '>=', '']:
            args = [cmpop, x, y]
            self.assertEqual(pyfunc(*args), cfunc(*args), msg='failed on {}'.format(args))