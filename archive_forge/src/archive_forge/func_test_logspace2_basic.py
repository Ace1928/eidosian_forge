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
def test_logspace2_basic(self):

    def inputs():
        yield (1, 60)
        yield (-1, 60)
        yield (-60, -1)
        yield (-1, -60)
        yield (60, -1)
        yield (1.0, 60.0)
        yield (-60.0, -1.0)
        yield (-1.0, 60.0)
        yield (0.0, np.e)
        yield (0.0, np.pi)
        yield (np.complex64(1), np.complex64(2))
        yield (np.complex64(2j), np.complex64(4j))
        yield (np.complex64(2), np.complex64(4j))
        yield (np.complex64(1 + 2j), np.complex64(3 + 4j))
        yield (np.complex64(1 - 2j), np.complex64(3 - 4j))
        yield (np.complex64(-1 + 2j), np.complex64(3 + 4j))
    pyfunc = logspace2
    cfunc = jit(nopython=True)(pyfunc)
    for start, stop in inputs():
        np.testing.assert_allclose(pyfunc(start, stop), cfunc(start, stop))