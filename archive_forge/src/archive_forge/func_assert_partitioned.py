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
def assert_partitioned(self, pyfunc, cfunc, d, kth):
    prev = 0
    for k in np.sort(kth):
        np.testing.assert_array_less(d[prev:k], d[k], err_msg='kth %d' % k)
        self.assertTrue((d[k:] >= d[k]).all(), msg='kth %d, %r not greater equal %d' % (k, d[k:], d[k]))
        prev = k + 1
        self.partition_sanity_check(pyfunc, cfunc, d, k)