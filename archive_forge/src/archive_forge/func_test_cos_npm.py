import itertools
import math
import sys
from numba import jit, types
from numba.tests.support import TestCase
from .complex_usecases import *
import unittest
def test_cos_npm(self):
    self.check_unary_func(cos_usecase, no_pyobj_flags, ulps=2)