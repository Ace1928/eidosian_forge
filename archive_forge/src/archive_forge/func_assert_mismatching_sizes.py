import contextlib
import gc
from itertools import product, cycle
import sys
import warnings
from numbers import Number, Integral
import platform
import numpy as np
from numba import jit, njit, typeof
from numba.core import errors
from numba.tests.support import (TestCase, tag, needs_lapack, needs_blas,
from .matmul_usecase import matmul_usecase
import unittest
def assert_mismatching_sizes(self, cfunc, args, is_out=False):
    with self.assertRaises(ValueError) as raises:
        cfunc(*args)
    msg = 'incompatible output array size' if is_out else 'incompatible array sizes'
    self.assertIn(msg, str(raises.exception))