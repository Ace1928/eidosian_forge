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
def test_indices_basic(self):
    pyfunc = np_indices
    cfunc = njit(np_indices)

    def inputs():
        yield (4, 3)
        yield (4,)
        yield (0,)
        yield (2, 2, 3, 5)
    for dims in inputs():
        self.assertPreciseEqual(pyfunc(dims), cfunc(dims))