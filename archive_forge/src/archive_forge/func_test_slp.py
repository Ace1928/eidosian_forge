import platform
import numpy as np
from numba import types
import unittest
from numba import njit
from numba.core import config
from numba.tests.support import TestCase
@TestCase.run_test_in_subprocess(envvars={'NUMBA_SLP_VECTORIZE': '1', **_skylake_env})
def test_slp(self):

    def foo(a1, a2, b1, b2, A):
        A[0] = a1 * (a1 + b1)
        A[1] = a2 * (a2 + b2)
        A[2] = a1 * (a1 + b1)
        A[3] = a2 * (a2 + b2)
    ty = types.float64
    llvm_ir = self.gen_ir(foo, (ty,) * 4 + (ty[::1],), fastmath=True)
    self.assertIn('2 x double', llvm_ir)