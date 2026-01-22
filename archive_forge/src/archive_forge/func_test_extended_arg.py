import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import opcode
import sys
import textwrap
import types
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.concrete import OFFSET_AS_INSTRUCTION
from _pydevd_frame_eval.vendored.bytecode.tests import get_code, TestCase
def test_extended_arg(self):
    co_code = b'\x90\x12\x904\x90\xabd\xcd'
    code = get_code('x=1')
    args = (code.co_argcount,) if sys.version_info < (3, 8) else (code.co_argcount, code.co_posonlyargcount)
    args += (code.co_kwonlyargcount, code.co_nlocals, code.co_stacksize, code.co_flags, co_code, code.co_consts, code.co_names, code.co_varnames, code.co_filename, code.co_name, code.co_firstlineno, code.co_linetable if sys.version_info >= (3, 10) else code.co_lnotab, code.co_freevars, code.co_cellvars)
    code = types.CodeType(*args)
    bytecode = ConcreteBytecode.from_code(code)
    self.assertListEqual(list(bytecode), [ConcreteInstr('LOAD_CONST', 305441741, lineno=1)])
    bytecode = ConcreteBytecode.from_code(code, extended_arg=True)
    expected = [ConcreteInstr('EXTENDED_ARG', 18, lineno=1), ConcreteInstr('EXTENDED_ARG', 52, lineno=1), ConcreteInstr('EXTENDED_ARG', 171, lineno=1), ConcreteInstr('LOAD_CONST', 205, lineno=1)]
    self.assertListEqual(list(bytecode), expected)