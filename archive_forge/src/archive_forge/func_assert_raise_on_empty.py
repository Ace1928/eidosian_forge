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
def assert_raise_on_empty(self, cfunc, args):
    msg = 'Arrays cannot be empty'
    self.assert_error(cfunc, args, msg, np.linalg.LinAlgError)