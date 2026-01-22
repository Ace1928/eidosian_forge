import array
import numpy as np
from numba import jit, njit
import numba.core.typing.cffi_utils as cffi_support
from numba.core import types, errors
from numba.tests.support import TestCase, skip_unless_cffi
import numba.tests.cffi_usecases as mod
import unittest
def check_vector_sin(self, cfunc, x, y):
    cfunc(x, y)
    np.testing.assert_allclose(y, np.sin(x))