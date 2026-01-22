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
def test_argpartition_boolean_inputs(self):
    pyfunc = argpartition
    cfunc = jit(nopython=True)(pyfunc)
    for d in (np.linspace(1, 10, 17), np.array((True, False, True))):
        for kth in (True, False, -1, 0, 1):
            self.argpartition_sanity_check(pyfunc, cfunc, d, kth)