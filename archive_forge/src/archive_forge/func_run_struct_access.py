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
@contextlib.contextmanager
def run_struct_access(self, struct_class, buf, offset=0):
    with self.compile_function(1) as (context, builder, args, call):
        inst = struct_class(context, builder)
        sptr = builder.add(args[0], machine_const(offset))
        sptr = builder.inttoptr(sptr, ir.PointerType(inst._type))
        inst = struct_class(context, builder, ref=sptr)
        yield (context, builder, args, inst)
        builder.ret(ir.Constant(machine_int, 0))
    call(self.get_bytearray_addr(buf))