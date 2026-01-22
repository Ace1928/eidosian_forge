import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def test_closure4(self):

    @njit
    def foo(x):

        def bar(z):
            if z < 0:
                raise ValueError('exception')
            return z
        try:
            return [x for x in map(bar, [1, 2, 3, x])]
        except:
            print('CAUGHT')
    with captured_stdout() as stdout:
        res = foo(-1)
    self.assertEqual(stdout.getvalue().strip(), 'CAUGHT')
    self.assertIsNone(res)
    with captured_stdout() as stdout:
        res = foo(4)
    self.assertEqual(stdout.getvalue(), '')
    self.assertEqual(res, [1, 2, 3, 4])