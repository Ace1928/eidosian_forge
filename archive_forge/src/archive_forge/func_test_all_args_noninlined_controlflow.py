import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
@skip_unless_py10_or_later
def test_all_args_noninlined_controlflow(self):
    """
        Tests generating large args when one of the inputs
        has the change suggested in the error message
        for inlined control flow.
        """

    def inline_func(flag):
        a_val = 1 if flag else 2
        return sum_jit_func(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, a_val, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    py_func = inline_func
    cfunc = njit()(inline_func)
    a = py_func(False)
    b = cfunc(False)
    self.assertEqual(a, b)