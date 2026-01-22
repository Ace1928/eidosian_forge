import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
@skip_unless_py10_or_later
def test_large_const_dict_inline_controlflow(self):
    """
        Tests generating a large dictionary when one of
        the inputs requires inline control flow
        has the change suggested in the error message
        for inlined control flow.
        """

    def inline_func(a, flag):
        d = {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'H': 1 if flag else 2, 'I': 1, 'J': 1, 'K': 1, 'L': 1, 'M': 1, 'N': 1, 'O': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': a}
        return d['S']
    with self.assertRaises(UnsupportedError) as raises:
        njit()(inline_func)('a_string', False)
    self.assertIn('You can resolve this issue by moving the control flow out', str(raises.exception))