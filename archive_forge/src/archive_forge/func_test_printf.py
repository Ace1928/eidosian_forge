import contextlib
import ctypes
import struct
import sys
import llvmlite.ir as ir
import numpy as np
import unittest
from numba.core import types, typing, cgutils, cpu
from numba.core.compiler_lock import global_compiler_lock
from numba.tests.support import TestCase, run_in_subprocess
def test_printf(self):
    value = 123456
    code = f'if 1:\n        from numba import njit, types\n        from numba.extending import intrinsic\n\n        @intrinsic\n        def printf(tyctx, int_arg):\n            sig = types.void(int_arg)\n            def codegen(cgctx, builder, sig, llargs):\n                cgctx.printf(builder, "%d\\n", *llargs)\n            return sig, codegen\n\n        @njit\n        def foo():\n            printf({value})\n\n        foo()\n        '
    out, _ = run_in_subprocess(code)
    self.assertIn(str(value), out.decode())