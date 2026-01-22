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
def test_disabled_compilation_nested_call(self):

    @jit(['(intp,)'])
    def foo(a):
        return a

    @jit
    def bar():
        foo(1)
        foo(np.ones(1))
    with self.assertRaises(errors.TypingError) as raises:
        bar()
    m = '.*Invalid use of.*with parameters \\(array\\(float64, 1d, C\\)\\).*'
    self.assertRegex(str(raises.exception), m)