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
def test_iter_next(self, flags=forceobj_flags):
    pyfunc = iter_next_usecase
    cfunc = jit((types.UniTuple(types.int32, 3),), **flags)(pyfunc)
    self.assertPreciseEqual(cfunc((1, 42, 5)), (1, 42))
    cfunc = jit((types.UniTuple(types.int32, 1),), **flags)(pyfunc)
    with self.assertRaises(StopIteration):
        cfunc((1,))