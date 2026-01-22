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
def test_iinfo(self):
    types = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
    attrs = ('min', 'max', 'bits')
    for ty in types:
        self.check(iinfo, attrs, ty(1))
        hc_func = self.create_harcoded_variant(np.iinfo, ty)
        self.check(hc_func, attrs)
    with self.assertTypingError():
        cfunc = jit(nopython=True)(iinfo)
        cfunc(np.float64(7))