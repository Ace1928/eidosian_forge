import unittest
from numba import jit
from numba.core import types
def test_long_int_return_type(self, flags=force_pyobj_flags):
    self.test_int_return_type(flags=flags, operands=(2 ** 33, 2 ** 40))