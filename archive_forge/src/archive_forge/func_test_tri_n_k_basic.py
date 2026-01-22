import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
def test_tri_n_k_basic(self):
    pyfunc = tri_n_k
    cfunc = jit(nopython=True)(pyfunc)
    _check = partial(self._check_output, pyfunc, cfunc)

    def n_variations():
        return np.arange(-4, 8)

    def k_variations():
        return np.arange(-10, 10)
    for n in n_variations():
        params = {'N': n}
        _check(params)
    for n in n_variations():
        for k in k_variations():
            params = {'N': n, 'k': k}
            _check(params)