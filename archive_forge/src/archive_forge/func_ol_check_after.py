import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
@overload(check_after, prefer_literal=True)
def ol_check_after(d):
    nonlocal checked_after
    if not checked_after:
        checked_after = True
        self.assertTrue(isinstance(d, types.DictType))
        self.assertTrue(d.initial_value is None)
    return lambda d: None