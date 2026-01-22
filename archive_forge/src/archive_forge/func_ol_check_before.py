import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
@overload(check_before, prefer_literal=True)
def ol_check_before(d):
    nonlocal checked_before
    if not checked_before:
        checked_before = True
        a = {'a': 1, 'b': 2, 'c': 3}
        self.assertTrue(isinstance(d, types.DictType))
        self.assertEqual(d.initial_value, a)
    return lambda d: None