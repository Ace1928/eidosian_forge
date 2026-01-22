import itertools
import functools
import sys
import operator
from collections import namedtuple
import numpy as np
import unittest
import warnings
from numba import jit, typeof, njit, typed
from numba.core import errors, types, config
from numba.tests.support import (TestCase, tag, ignore_internal_warnings,
from numba.core.extending import overload_method, box
def test_python_numpy_scalar_alias_problem(self):

    @njit
    def foo():
        return isinstance(np.intp(10), int)
    self.assertEqual(foo(), True)
    self.assertEqual(foo.py_func(), False)

    @njit
    def bar():
        return isinstance(1, np.intp)
    self.assertEqual(bar(), True)
    self.assertEqual(bar.py_func(), False)