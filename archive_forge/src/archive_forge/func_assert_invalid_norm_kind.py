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
def assert_invalid_norm_kind(self, cfunc, args):
    """
        For use in norm() and cond() tests.
        """
    msg = 'Invalid norm order for matrices.'
    self.assert_error(cfunc, args, msg, ValueError)