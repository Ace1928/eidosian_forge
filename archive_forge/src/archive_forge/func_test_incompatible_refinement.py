import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def test_incompatible_refinement(self):

    @njit
    def udt():
        try:
            lst = typed.List()
            print('A')
            lst.append(0)
            print('B')
            lst.append('fda')
            print('C')
            return lst
        except Exception:
            print('D')
    with self.assertRaises(TypingError) as raises:
        udt()
    self.assertRegex(str(raises.exception), 'Cannot refine type|cannot safely cast unicode_type to int(32|64)')