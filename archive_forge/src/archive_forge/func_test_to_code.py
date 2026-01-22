import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import textwrap
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, FreeVar, Bytecode, SetLineno, ConcreteInstr
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, get_code
def test_to_code(self):
    code = Bytecode()
    code.first_lineno = 50
    code.extend([Instr('LOAD_NAME', 'print'), Instr('LOAD_CONST', '%s'), Instr('LOAD_GLOBAL', 'a'), Instr('BINARY_MODULO'), Instr('CALL_FUNCTION', 1), Instr('RETURN_VALUE')])
    co = code.to_code()
    self.assertEqual(co.co_stacksize, 3)
    co = code.to_code(stacksize=42)
    self.assertEqual(co.co_stacksize, 42)