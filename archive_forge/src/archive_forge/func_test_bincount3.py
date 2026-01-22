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
def test_bincount3(self):
    pyfunc = bincount3
    cfunc = jit(nopython=True)(pyfunc)
    for seq in self.bincount_sequences():
        a_max = max(seq)
        for minlength in (a_max, a_max + 2):
            expected = pyfunc(seq, None, minlength)
            got = cfunc(seq, None, minlength)
            self.assertEqual(len(expected), len(got))
            self.assertPreciseEqual(expected, got)