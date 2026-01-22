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
def chain_assign(fs, inner=ident):
    tab_head, tab_tail = (fs[-1], fs[:-1])

    @jit(nopython=True)
    def assign(out, x):
        inner(out, x)
        out[0] += tab_head(x)
    if tab_tail:
        return chain_assign(tab_tail, assign)
    else:
        return assign