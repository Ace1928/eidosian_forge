import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def test_stack_effect_error(self, flags=force_pyobj_jit_opt):
    pyfunc = stack_effect_error
    cfunc = jit((types.int32,), **flags)(pyfunc)
    for x in (0, 1, 2, 3):
        self.assertPreciseEqual(pyfunc(x), cfunc(x))