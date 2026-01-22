from numba import int32, int64
from numba import jit
from numba.core import types
from numba.extending import overload
from numba.tests.support import TestCase, tag
import unittest
def test_argcast(self):
    """
        Issue #1488: implicitly casting an argument variable should not
        break nested calls.
        """
    cfunc, check = self.compile_func(argcast)
    check(1, 0)
    check(1, 1)