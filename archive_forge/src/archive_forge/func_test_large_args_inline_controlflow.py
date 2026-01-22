import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
@skip_unless_py10_or_later
def test_large_args_inline_controlflow(self):
    """
        Tests generating large args when one of the inputs
        has inlined controlflow.
        """

    def inline_func(flag):
        return sum_jit_func(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 if flag else 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, arg41=1)
    with self.assertRaises(UnsupportedError) as raises:
        njit()(inline_func)(False)
    self.assertIn('You can resolve this issue by moving the control flow out', str(raises.exception))