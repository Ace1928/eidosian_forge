import itertools
import unittest
import numpy as np
from numba import jit, njit
from numba.core import types
from numba.tests import usecases
from numba.tests.support import TestCase
@TestCase.run_test_in_subprocess
def test_string_conversion(self):
    pyfunc = usecases.string_conversion
    cfunc = jit((types.int32,), forceobj=True)(pyfunc)
    self.assertEqual(pyfunc(1), cfunc(1))
    cfunc = jit((types.float32,), forceobj=True)(pyfunc)
    self.assertEqual(pyfunc(1.1), cfunc(1.1))