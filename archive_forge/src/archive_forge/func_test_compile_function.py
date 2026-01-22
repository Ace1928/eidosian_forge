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
def test_compile_function(self):
    with self.compile_function(2) as (context, builder, args, call):
        res = builder.add(args[0], args[1])
        builder.ret(res)
    self.assertEqual(call(5, -2), 3)
    self.assertEqual(call(4, 2), 6)