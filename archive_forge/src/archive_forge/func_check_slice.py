import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def check_slice(self, pyfunc):
    tup = (4, 5, 6, 7)
    cfunc = njit((types.UniTuple(types.int64, 4),))(pyfunc)
    self.assertPreciseEqual(cfunc(tup), pyfunc(tup))
    args = types.Tuple((types.int64, types.int32, types.int64, types.int32))
    cfunc = njit((args,))(pyfunc)
    self.assertPreciseEqual(cfunc(tup), pyfunc(tup))