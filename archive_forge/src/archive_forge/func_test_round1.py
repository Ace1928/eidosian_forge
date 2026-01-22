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
def test_round1(self, flags=forceobj_flags):
    pyfunc = round_usecase1
    for tp in (types.float64, types.float32):
        cfunc = jit((tp,), **flags)(pyfunc)
        values = [-1.6, -1.5, -1.4, -0.5, 0.0, 0.1, 0.5, 0.6, 1.4, 1.5, 5.0]
        values += [-0.1, -0.0]
        for x in values:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))