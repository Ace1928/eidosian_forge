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
def test_instruction_namer_pass(self):
    asm = asm_inlineasm3.format(triple=llvm.get_default_triple())
    mod = llvm.parse_assembly(asm)
    pm = llvm.ModulePassManager()
    pm.add_instruction_namer_pass()
    pm.run(mod)
    func = mod.get_function('foo')
    first_block = next(func.blocks)
    instructions = list(first_block.instructions)
    self.assertEqual(instructions[0].name, 'i')
    self.assertEqual(instructions[1].name, 'i2')