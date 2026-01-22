import array
import numpy as np
from numba import jit, njit
import numba.core.typing.cffi_utils as cffi_support
from numba.core import types, errors
from numba.tests.support import TestCase, skip_unless_cffi
import numba.tests.cffi_usecases as mod
import unittest
def test_user_defined_symbols(self):
    pyfunc = mod.use_user_defined_symbols
    cfunc = jit(nopython=True)(pyfunc)
    self.assertEqual(pyfunc(), cfunc())