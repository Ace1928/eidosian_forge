import multiprocessing
import platform
import threading
import pickle
import weakref
from itertools import chain
from io import StringIO
import numpy as np
from numba import njit, jit, typeof, vectorize
from numba.core import types, errors
from numba import _dispatcher
from numba.tests.support import TestCase, captured_stdout
from numba.np.numpy_support import as_dtype
from numba.core.dispatcher import Dispatcher
from numba.extending import overload
from numba.tests.support import needs_lapack, SerialMixin
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
import unittest
def test_dispatcher_in_sequence_arg(self):

    @jit(nopython=True)
    def one(x):
        return x + 1

    @jit(nopython=True)
    def two(x):
        return one(one(x))

    @jit(nopython=True)
    def three(x):
        return one(one(one(x)))

    @jit(nopython=True)
    def choose(fns, x):
        return (fns[0](x), fns[1](x), fns[2](x))
    self.assertEqual(choose((one, two, three), 1), (2, 3, 4))
    self.assertEqual(choose([one, one, one], 1), (2, 2, 2))