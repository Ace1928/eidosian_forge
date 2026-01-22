import unittest
import math
import sys
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, tag
def test_constant_uint64(self, nopython=False):
    pyfunc = usecase_uint64_constant
    self.check_nullary_func(pyfunc, nopython=nopython)