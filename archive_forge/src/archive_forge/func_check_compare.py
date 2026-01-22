import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def check_compare(self, cfunc, pyfunc, *args, **kwargs):
    with captured_stdout() as stdout:
        pyfunc(*args, **kwargs)
    expect = stdout.getvalue()
    with captured_stdout() as stdout:
        cfunc(*args, **kwargs)
    got = stdout.getvalue()
    self.assertEqual(expect, got, msg='args={} kwargs={}'.format(args, kwargs))