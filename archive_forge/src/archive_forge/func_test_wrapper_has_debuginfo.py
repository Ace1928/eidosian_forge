from numba.tests.support import override_config
from numba.cuda.testing import skip_on_cudasim
from numba import cuda
from numba.core import types
from numba.cuda.testing import CUDATestCase
import itertools
import re
import unittest
def test_wrapper_has_debuginfo(self):
    sig = (types.int32[::1],)

    @cuda.jit(sig, debug=True, opt=0)
    def f(x):
        x[0] = 1
    llvm_ir = f.inspect_llvm(sig)
    defines = [line for line in llvm_ir.splitlines() if 'define void @"_ZN6cudapy' in line]
    self.assertEqual(len(defines), 1)
    wrapper_define = defines[0]
    self.assertIn('!dbg', wrapper_define)