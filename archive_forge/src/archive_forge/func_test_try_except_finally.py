import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def test_try_except_finally(self):

    @njit
    def udt(p, q):
        try:
            print('A')
            if p:
                print('B')
                raise MyError
            print('C')
        except:
            print('D')
        finally:
            try:
                print('E')
                if q:
                    print('F')
                    raise MyError
            except Exception:
                print('G')
            else:
                print('H')
            finally:
                print('I')
    cases = list(product([True, False], repeat=2))
    self.assertTrue(cases)
    for p, q in cases:
        self.check_compare(udt, udt.py_func, p=p, q=q)