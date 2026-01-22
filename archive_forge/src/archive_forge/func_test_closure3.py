import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def test_closure3(self):

    @njit
    def foo(x):

        def bar(z):
            try:
                raise ValueError('exception')
            except:
                print('CAUGHT')
                return z
        return [x for x in map(bar, [1, 2, 3])]
    with captured_stdout() as stdout:
        res = foo(10)
    self.assertEqual(res, [1, 2, 3])
    self.assertEqual(stdout.getvalue().split(), ['CAUGHT'] * 3)