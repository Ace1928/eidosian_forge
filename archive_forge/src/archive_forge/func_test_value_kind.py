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
def test_value_kind(self):
    mod = self.module()
    self.assertEqual(mod.get_global_variable('glob').value_kind, llvm.ValueKind.global_variable)
    func = mod.get_function('sum')
    self.assertEqual(func.value_kind, llvm.ValueKind.function)
    block = list(func.blocks)[0]
    self.assertEqual(block.value_kind, llvm.ValueKind.basic_block)
    inst = list(block.instructions)[1]
    self.assertEqual(inst.value_kind, llvm.ValueKind.instruction)
    self.assertEqual(list(inst.operands)[0].value_kind, llvm.ValueKind.constant_int)
    self.assertEqual(list(inst.operands)[1].value_kind, llvm.ValueKind.instruction)
    iasm_func = self.module(asm_inlineasm).get_function('foo')
    iasm_inst = list(list(iasm_func.blocks)[0].instructions)[0]
    self.assertEqual(list(iasm_inst.operands)[0].value_kind, llvm.ValueKind.inline_asm)