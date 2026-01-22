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
def test_getattr_value_retty(self):

    @njit
    def foo(x):
        return getattr(x, 'ndim')
    for x in range(3):
        tmp = np.empty((1,) * x)
        self.assertPreciseEqual(foo(tmp), foo.py_func(tmp))