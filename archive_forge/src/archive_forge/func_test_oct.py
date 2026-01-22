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
def test_oct(self, flags=forceobj_flags):
    pyfunc = oct_usecase
    cfunc = jit((types.int32,), **flags)(pyfunc)
    for x in [-8, -1, 0, 1, 8]:
        self.assertPreciseEqual(cfunc(x), pyfunc(x))