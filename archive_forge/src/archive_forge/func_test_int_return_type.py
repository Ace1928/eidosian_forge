import unittest
from numba import jit
from numba.core import types
def test_int_return_type(self, flags=force_pyobj_flags, int_type=types.int64, operands=(3, 4)):
    pyfunc = return_int
    cfunc = jit((int_type, int_type), **flags)(pyfunc)
    expected = pyfunc(*operands)
    got = cfunc(*operands)
    self.assertIs(type(got), type(expected))
    self.assertEqual(got, expected)