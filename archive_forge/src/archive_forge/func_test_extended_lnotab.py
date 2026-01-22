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
def test_extended_lnotab(self):
    concrete = ConcreteBytecode([ConcreteInstr('LOAD_CONST', 0), SetLineno(1 + 128), ConcreteInstr('STORE_NAME', 0), SetLineno(1 + 129), ConcreteInstr('LOAD_CONST', 1), SetLineno(1), ConcreteInstr('STORE_NAME', 1)])
    concrete.consts = [7, 8]
    concrete.names = ['x', 'y']
    concrete.first_lineno = 1
    code = concrete.to_code()
    expected = b'd\x00Z\x00d\x01Z\x01'
    self.assertEqual(code.co_code, expected)
    self.assertEqual(code.co_firstlineno, 1)
    self.assertEqual(code.co_lnotab, b'\x02\x7f\x00\x01\x02\x01\x02\x80\x00\xff')