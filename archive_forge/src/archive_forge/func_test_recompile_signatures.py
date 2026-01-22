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
def test_recompile_signatures(self):
    closure = 1

    @jit('int32(int32)')
    def foo(x):
        return x + closure
    self.assertPreciseEqual(foo(1), 2)
    self.assertPreciseEqual(foo(1.5), 2)
    closure = 2
    self.assertPreciseEqual(foo(1), 2)
    foo.recompile()
    self.assertPreciseEqual(foo(1), 3)
    self.assertPreciseEqual(foo(1.5), 3)