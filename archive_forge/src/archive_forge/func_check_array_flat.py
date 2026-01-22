import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def check_array_flat(self, arr, arrty=None):
    out = np.zeros(arr.size, dtype=arr.dtype)
    nb_out = out.copy()
    if arrty is None:
        arrty = typeof(arr)
    cfunc = njit((arrty, typeof(out)))(array_flat)
    array_flat(arr, out)
    cfunc(arr, nb_out)
    self.assertPreciseEqual(out, nb_out)