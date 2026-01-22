import array
import numpy as np
from numba import jit, njit
import numba.core.typing.cffi_utils as cffi_support
from numba.core import types, errors
from numba.tests.support import TestCase, skip_unless_cffi
import numba.tests.cffi_usecases as mod
import unittest
def test_type_map(self):
    signature = cffi_support.map_type(mod.ffi.typeof(mod.cffi_sin))
    self.assertEqual(len(signature.args), 1)
    self.assertEqual(signature.args[0], types.double)