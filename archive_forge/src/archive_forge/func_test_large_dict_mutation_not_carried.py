import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
@skip_unless_py10_or_later
def test_large_dict_mutation_not_carried(self):
    """Checks that the optimization for large dictionaries
        do not incorrectly update initial values due to other
        mutations.
        """

    def bar(d):
        ...

    @overload(bar)
    def ol_bar(d):
        a = {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'H': 1, 'I': 1, 'J': 1, 'K': 1, 'L': 1, 'M': 1, 'N': 1, 'O': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 7}
        if d.initial_value is None:
            return lambda d: literally(d)
        self.assertTrue(isinstance(d, types.DictType))
        self.assertEqual(d.initial_value, a)
        return lambda d: d

    @njit
    def foo():
        d = {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'H': 1, 'I': 1, 'J': 1, 'K': 1, 'L': 1, 'M': 1, 'N': 1, 'O': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 7}
        d['X'] = 4
        bar(d)
    foo()