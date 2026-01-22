import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def test_save_caught(self):

    @njit
    def udt(x):
        try:
            if x:
                raise ZeroDivisionError
            r = 123
        except Exception as e:
            r = 321
            return r
        return r
    with self.assertRaises(UnsupportedError) as raises:
        udt(True)
    self.assertIn('Exception object cannot be stored into variable (e)', str(raises.exception))