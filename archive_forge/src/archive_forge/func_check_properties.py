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
def check_properties(arr, layout, aligned):
    self.assertEqual(arr.flags.aligned, aligned)
    if layout == 'C':
        self.assertEqual(arr.flags.c_contiguous, True)
    if layout == 'F':
        self.assertEqual(arr.flags.f_contiguous, True)