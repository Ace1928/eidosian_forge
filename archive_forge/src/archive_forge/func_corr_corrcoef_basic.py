import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
def corr_corrcoef_basic(self, pyfunc, first_arg_name):
    cfunc = jit(nopython=True)(pyfunc)
    _check = partial(self._check_output, pyfunc, cfunc, abs_tol=1e-14)

    def input_variations():
        yield np.array([[0, 2], [1, 1], [2, 0]]).T
        yield self.rnd.randn(100).reshape(5, 20)
        yield np.asfortranarray(np.array([[0, 2], [1, 1], [2, 0]]).T)
        yield self.rnd.randn(100).reshape(5, 20)[:, ::2]
        yield np.array([0.3942, 0.5969, 0.773, 0.9918, 0.7964])
        yield np.full((4, 5), fill_value=True)
        yield np.array([np.nan, 0.5969, -np.inf, 0.9918, 0.7964])
        yield np.linspace(-3, 3, 33).reshape(33, 1)
        yield ((0.1, 0.2), (0.11, 0.19), (0.09, 0.21))
        yield ((0.1, 0.2), (0.11, 0.19), (0.09j, 0.21j))
        yield (-2.1, -1, 4.3)
        yield (1, 2, 3)
        yield [4, 5, 6]
        yield ((0.1, 0.2, 0.3), (0.1, 0.2, 0.3))
        yield [(1, 2, 3), (1, 3, 2)]
        yield 3.142
        yield ((1.1, 2.2, 1.5),)
        yield np.array([])
        yield np.array([]).reshape(0, 2)
        yield np.array([]).reshape(2, 0)
        yield ()
    for input_arr in input_variations():
        _check({first_arg_name: input_arr})