import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def test_try_inner_raise(self):

    @njit
    def inner(x):
        if x:
            raise MyError

    @njit
    def udt(x):
        try:
            inner(x)
            return 'not raised'
        except:
            return 'caught'
    self.assertEqual(udt(False), 'not raised')
    self.assertEqual(udt(True), 'caught')