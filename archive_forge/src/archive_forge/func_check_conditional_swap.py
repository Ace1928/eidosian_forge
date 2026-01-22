import numpy as np
import unittest
from numba import jit, njit
from numba.core import errors, types
from numba import typeof
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.tests.support import no_pyobj_flags as nullary_no_pyobj_flags
from numba.tests.support import force_pyobj_flags as nullary_force_pyobj_flags
def check_conditional_swap(self, flags=force_pyobj_flags):
    cfunc = jit((types.int32, types.int32), **flags)(conditional_swap)
    self.assertPreciseEqual(cfunc(4, 5), (5, 4))
    self.assertPreciseEqual(cfunc(0, 5), (0, 5))