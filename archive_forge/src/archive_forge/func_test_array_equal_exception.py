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
def test_array_equal_exception(self):
    pyfunc = array_equal
    cfunc = jit(nopython=True)(pyfunc)
    with self.assertRaises(TypingError) as raises:
        cfunc(np.arange(3 * 4).reshape(3, 4), None)
    self.assertIn('Both arguments to "array_equals" must be array-like', str(raises.exception))