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
def test_dispatcher_as_arg_usecase(self):

    @jit(nopython=True)
    def maximum(seq, cmpfn):
        tmp = seq[0]
        for each in seq[1:]:
            cmpval = cmpfn(tmp, each)
            if cmpval < 0:
                tmp = each
        return tmp
    got = maximum([1, 2, 3, 4], cmpfn=jit(lambda x, y: x - y))
    self.assertEqual(got, 4)
    got = maximum(list(zip(range(5), range(5)[::-1])), cmpfn=jit(lambda x, y: x[0] - y[0]))
    self.assertEqual(got, (4, 0))
    got = maximum(list(zip(range(5), range(5)[::-1])), cmpfn=jit(lambda x, y: x[1] - y[1]))
    self.assertEqual(got, (0, 4))