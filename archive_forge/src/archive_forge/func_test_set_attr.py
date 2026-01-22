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
def test_set_attr(self):
    instr = ConcreteInstr('LOAD_CONST', 5, lineno=12)
    instr.name = 'LOAD_FAST'
    self.assertEqual(instr.name, 'LOAD_FAST')
    self.assertEqual(instr.opcode, 124)
    self.assertRaises(TypeError, setattr, instr, 'name', 3)
    self.assertRaises(ValueError, setattr, instr, 'name', 'xxx')
    instr.opcode = 100
    self.assertEqual(instr.name, 'LOAD_CONST')
    self.assertEqual(instr.opcode, 100)
    self.assertRaises(ValueError, setattr, instr, 'opcode', -12)
    self.assertRaises(TypeError, setattr, instr, 'opcode', 'abc')
    instr.arg = 305441741
    self.assertEqual(instr.arg, 305441741)
    self.assertEqual(instr.size, 8)
    instr.arg = 0
    self.assertEqual(instr.arg, 0)
    self.assertEqual(instr.size, 2)
    self.assertRaises(ValueError, setattr, instr, 'arg', -1)
    self.assertRaises(ValueError, setattr, instr, 'arg', 2147483647 + 1)
    self.assertRaises(AttributeError, setattr, instr, 'size', 3)
    instr.lineno = 33
    self.assertEqual(instr.lineno, 33)
    self.assertRaises(TypeError, setattr, instr, 'lineno', 1.0)
    self.assertRaises(ValueError, setattr, instr, 'lineno', -1)