import ctypes
import threading
from ctypes import CFUNCTYPE, c_int, c_int32
from ctypes.util import find_library
import gc
import locale
import os
import platform
import re
import subprocess
import sys
import unittest
from contextlib import contextmanager
from tempfile import mkstemp
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.binding import ffi
from llvmlite.tests import TestCase
def fpadd_ll_module(self):
    f64 = ir.DoubleType()
    f32 = ir.FloatType()
    fnty = ir.FunctionType(f64, (f32, f64))
    module = ir.Module()
    func = ir.Function(module, fnty, name='fpadd')
    block = func.append_basic_block()
    builder = ir.IRBuilder(block)
    a, b = func.args
    arg0 = builder.fpext(a, f64)
    result = builder.fadd(arg0, b)
    builder.ret(result)
    llmod = llvm.parse_assembly(str(module))
    llmod.verify()
    return llmod