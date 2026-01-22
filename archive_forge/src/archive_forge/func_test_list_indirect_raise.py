import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def test_list_indirect_raise(self):

    @njit
    def appender(lst, n, raise_at):
        for i in range(n):
            if i == raise_at:
                raise IndexError
            lst.append(i)
        return lst

    @njit
    def udt(n, raise_at):
        lst = typed.List()
        lst.append(48657)
        try:
            appender(lst, n, raise_at)
        except Exception:
            return lst
        else:
            return lst
    out = udt(10, raise_at=5)
    self.assertEqual(list(out), [48657] + list(range(5)))
    out = udt(10, raise_at=10)
    self.assertEqual(list(out), [48657] + list(range(10)))