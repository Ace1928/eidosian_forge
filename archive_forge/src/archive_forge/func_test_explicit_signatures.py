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
def test_explicit_signatures(self):
    f = jit('(int64,int64)')(add)
    self.assertPreciseEqual(f(1.5, 2.5), 3)
    self.assertEqual(len(f.overloads), 1, f.overloads)
    f = jit(['(int64,int64)', '(float64,float64)'])(add)
    self.assertPreciseEqual(f(1, 2), 3)
    self.assertPreciseEqual(f(1.5, 2.5), 4.0)
    self.assertPreciseEqual(f(np.int32(1), 2.5), 3.5)
    with self.assertRaises(TypeError) as cm:
        f(1j, 1j)
    self.assertIn('No matching definition', str(cm.exception))
    self.assertEqual(len(f.overloads), 2, f.overloads)
    f = jit(['(float32,float32)', '(float64,float64)'])(add)
    self.assertPreciseEqual(f(np.float32(1), np.float32(2 ** (-25))), 1.0)
    self.assertPreciseEqual(f(1, 2 ** (-25)), 1.0000000298023224)
    f = jit(['(float32,float64)', '(float64,float32)', '(int64,int64)'])(add)
    with self.assertRaises(TypeError) as cm:
        f(1.0, 2.0)
    self.assertRegex(str(cm.exception), 'Ambiguous overloading for <function add [^>]*> \\(float64, float64\\):\\n\\(float32, float64\\) -> float64\\n\\(float64, float32\\) -> float64')
    self.assertNotIn('int64', str(cm.exception))