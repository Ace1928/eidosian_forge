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
def test_tri_exceptions(self):
    pyfunc = tri_n_m_k
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()

    def _check(k):
        with self.assertTypingError() as raises:
            cfunc(5, 6, k=k)
        assert 'k must be an integer' in str(raises.exception)
    for k in (1.5, True, np.inf, [1, 2]):
        _check(k)