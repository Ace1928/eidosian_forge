import numpy as np
import unittest
from numba import jit, njit
from numba.core import errors, types
from numba import typeof
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.tests.support import no_pyobj_flags as nullary_no_pyobj_flags
from numba.tests.support import force_pyobj_flags as nullary_force_pyobj_flags
def unpack_nrt():
    a = np.zeros(1)
    b = np.zeros(2)
    tup = (b, a)
    alpha, beta = tup
    return (alpha, beta)