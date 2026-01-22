import math
import numpy as np
from numba.tests.support import captured_stdout, override_config
from numba import njit, vectorize, guvectorize
import unittest
def test_jit_subset_code(self):

    def foo(x):
        return x + math.sin(x)
    fastfoo = njit(fastmath={'reassoc', 'nsz'})(foo)
    slowfoo = njit()(foo)
    self.assertEqual(fastfoo(0.5), slowfoo(0.5))
    fastllvm = fastfoo.inspect_llvm(fastfoo.signatures[0])
    slowllvm = slowfoo.inspect_llvm(slowfoo.signatures[0])
    self.assertNotIn('fadd fast', slowllvm)
    self.assertNotIn('call fast', slowllvm)
    self.assertNotIn('fadd reassoc nsz', slowllvm)
    self.assertNotIn('call reassoc nsz', slowllvm)
    self.assertNotIn('fadd nsz reassoc', slowllvm)
    self.assertNotIn('call nsz reassoc', slowllvm)
    self.assertTrue('fadd nsz reassoc' in fastllvm or 'fadd reassoc nsz' in fastllvm, fastllvm)
    self.assertTrue('call nsz reassoc' in fastllvm or 'call reassoc nsz' in fastllvm, fastllvm)