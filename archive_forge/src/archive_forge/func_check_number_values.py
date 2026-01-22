import sys
from itertools import product
import numpy as np
import unittest
from numba.core import types
from numba.core.errors import NumbaNotImplementedError
from numba.tests.support import TestCase
from numba.tests.enum_usecases import Shake, RequestError
from numba.np import numpy_support
def check_number_values(self, func):
    """
        Test *func*() with scalar numeric values.
        """
    f = func
    self.assertIn(f(1), (types.int32, types.int64))
    self.assertIn(f(2 ** 31 - 1), (types.int32, types.int64))
    self.assertIn(f(-2 ** 31), (types.int32, types.int64))
    self.assertIs(f(1.0), types.float64)
    self.assertIs(f(1j), types.complex128)
    self.assertIs(f(True), types.bool_)
    self.assertIs(f(False), types.bool_)
    for name in ('int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'intc', 'uintc', 'intp', 'uintp', 'float32', 'float64', 'complex64', 'complex128', 'bool_'):
        val = getattr(np, name)()
        self.assertIs(f(val), getattr(types, name))