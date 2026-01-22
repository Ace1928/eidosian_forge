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
def test_getattr_func_retty(self):

    @njit
    def foo(x):
        attr = getattr(x, '__hash__')
        return attr()
    for x in (1, 2.34, (5, 6, 7)):
        self.assertPreciseEqual(foo(x), foo.py_func(x))