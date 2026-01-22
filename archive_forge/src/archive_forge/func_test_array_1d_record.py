import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np import numpy_support
def test_array_1d_record(self, flags=force_pyobj_flags):
    pyfunc = record_iter_usecase
    item_type = numpy_support.from_dtype(record_dtype)
    cfunc = jit((types.Array(item_type, 1, 'A'),), **flags)(pyfunc)
    arr = np.recarray(3, dtype=record_dtype)
    for i in range(3):
        arr[i].a = float(i * 2)
        arr[i].b = i + 2
    got = pyfunc(arr)
    self.assertPreciseEqual(cfunc(arr), got)