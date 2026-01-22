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
def test_type_unary(self):
    pyfunc = type_unary_usecase
    cfunc = jit(nopython=True)(pyfunc)

    def check(*args):
        expected = pyfunc(*args)
        self.assertPreciseEqual(cfunc(*args), expected)
    check(1.5, 2)
    check(1, 2.5)
    check(1.5j, 2)
    check(True, 2)
    check(2.5j, False)