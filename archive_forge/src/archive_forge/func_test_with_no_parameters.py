import gc
import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
from numba.np import numpy_support
def test_with_no_parameters(self):

    def f():
        pass
    self.assertEqual(f(), jit('()', nopython=True)(f)())