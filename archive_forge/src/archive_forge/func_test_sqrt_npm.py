import itertools
import math
import sys
from numba import jit, types
from numba.tests.support import TestCase
from .complex_usecases import *
import unittest
def test_sqrt_npm(self):
    self.check_unary_func(sqrt_usecase, no_pyobj_flags)
    values = [-10 ** i for i in range(36, 41)]
    self.run_unary(sqrt_usecase, [types.complex128], values, flags=no_pyobj_flags)