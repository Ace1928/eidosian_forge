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
def check_minmax_3(self, pyfunc, flags):

    def check(argty):
        cfunc = jit((argty,), **flags)(pyfunc)
        tup = (1.5, float('nan'), 2.5)
        for val in [tup, tup[::-1]]:
            self.assertPreciseEqual(cfunc(val), pyfunc(val))
    check(types.UniTuple(types.float64, 3))
    check(types.Tuple((types.float32, types.float64, types.float32)))