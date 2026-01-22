import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def mkfoo(a_, b_):

    def foo(choose_left):
        if choose_left:
            r = a_(1)
        else:
            r = b_(2)
        return r
    return foo