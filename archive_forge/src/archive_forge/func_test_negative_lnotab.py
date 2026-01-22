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
def test_negative_lnotab(self):
    concrete = ConcreteBytecode([ConcreteInstr('LOAD_CONST', 0), ConcreteInstr('STORE_NAME', 0), SetLineno(2), ConcreteInstr('LOAD_CONST', 1), ConcreteInstr('STORE_NAME', 1)])
    concrete.consts = [7, 8]
    concrete.names = ['x', 'y']
    concrete.first_lineno = 5
    code = concrete.to_code()
    expected = b'd\x00Z\x00d\x01Z\x01'
    self.assertEqual(code.co_code, expected)
    self.assertEqual(code.co_firstlineno, 5)
    self.assertEqual(code.co_lnotab, b'\x04\xfd')