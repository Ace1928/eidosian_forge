import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
@expected_failure_py311
@expected_failure_py312
def test_objmode_output_type(self):

    def bar(x):
        return np.asarray(list(reversed(x.tolist())))

    @njit
    def test_objmode():
        x = np.arange(5)
        y = np.zeros_like(x)
        try:
            with objmode(y='intp[:]'):
                y += bar(x)
        except Exception:
            pass
        return y
    with self.assertRaises(CompilerError) as raises:
        test_objmode()
    msg = 'unsupported control flow: with-context contains branches (i.e. break/return/raise) that can leave the block '
    self.assertIn(msg, str(raises.exception))