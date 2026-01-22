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
@global_compiler_lock
def call_func(*args):
    codegen = self.context.codegen()
    library = codegen.create_library('test_module.%s' % self.id())
    library.add_ir_module(module)
    cptr = library.get_pointer_to_function(function.name)
    cfunc = ctypes_fnty(cptr)
    return cfunc(*args)