import numpy as np
import unittest
from numba import jit, njit
from numba.core import errors, types
from numba import typeof
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.tests.support import no_pyobj_flags as nullary_no_pyobj_flags
from numba.tests.support import force_pyobj_flags as nullary_force_pyobj_flags
def test_unpack_shape_npm(self):
    pyfunc = unpack_shape
    cfunc = njit((types.Array(dtype=types.int32, ndim=3, layout='C'),))(pyfunc)
    a = np.zeros(shape=(1, 2, 3)).astype(np.int32)
    self.assertPreciseEqual(cfunc(a), pyfunc(a))