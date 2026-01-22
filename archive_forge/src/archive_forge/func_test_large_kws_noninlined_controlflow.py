import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
@skip_unless_py10_or_later
def test_large_kws_noninlined_controlflow(self):
    """
        Tests generating large kws when one of the inputs
        has the change suggested in the error message
        for inlined control flow.
        """

    def inline_func(flag):
        a_val = 1 if flag else 2
        return sum_jit_func(arg0=1, arg1=1, arg2=1, arg3=1, arg4=1, arg5=1, arg6=1, arg7=1, arg8=1, arg9=1, arg10=1, arg11=1, arg12=1, arg13=1, arg14=1, arg15=a_val)
    py_func = inline_func
    cfunc = njit()(inline_func)
    a = py_func(False)
    b = cfunc(False)
    self.assertEqual(a, b)