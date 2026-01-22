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
def test_getattr_with_default(self):

    @njit
    def foo(x, default):
        return getattr(x, '__not_a_valid_attr__', default)
    for x, y in zip((1, 2.34, (5, 6, 7)), (None, 20, 'some_string')):
        self.assertPreciseEqual(foo(x, y), foo.py_func(x, y))