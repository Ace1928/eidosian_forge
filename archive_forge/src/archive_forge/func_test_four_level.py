import math
import warnings
from numba import jit
from numba.core.errors import TypingError, NumbaWarning
from numba.tests.support import TestCase
import unittest
def test_four_level(self):
    from numba.tests.recursion_usecases import make_four_level
    pfunc = make_four_level()
    cfunc = make_four_level(jit(nopython=True))
    arg = 7
    self.assertPreciseEqual(pfunc(arg), cfunc(arg))