import itertools
import math
import sys
from numba import jit, types
from numba.tests.support import TestCase
from .complex_usecases import *
import unittest
def test_sinh_npm(self):
    self.check_unary_func(sinh_usecase, no_pyobj_flags, abs_tol='eps')