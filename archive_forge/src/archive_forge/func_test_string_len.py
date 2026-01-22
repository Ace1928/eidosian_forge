import itertools
import unittest
import numpy as np
from numba import jit, njit
from numba.core import types
from numba.tests import usecases
from numba.tests.support import TestCase
@TestCase.run_test_in_subprocess
def test_string_len(self):
    pyfunc = usecases.string_len
    cfunc = jit((types.pyobject,), forceobj=True)(pyfunc)
    test_str = '123456'
    self.assertEqual(pyfunc(test_str), cfunc(test_str))
    test_str = '1'
    self.assertEqual(pyfunc(test_str), cfunc(test_str))
    test_str = ''
    self.assertEqual(pyfunc(test_str), cfunc(test_str))