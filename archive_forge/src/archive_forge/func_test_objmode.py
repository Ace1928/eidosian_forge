import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
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