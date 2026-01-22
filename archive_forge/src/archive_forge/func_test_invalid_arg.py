import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import opcode
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase
def test_invalid_arg(self):
    label = Label()
    block = BasicBlock()
    self.assertRaises(ValueError, Instr, 'EXTENDED_ARG', 0)
    self.assertRaises(TypeError, Instr, 'JUMP_ABSOLUTE', 1)
    self.assertRaises(TypeError, Instr, 'JUMP_ABSOLUTE', 1.0)
    Instr('JUMP_ABSOLUTE', label)
    Instr('JUMP_ABSOLUTE', block)
    self.assertRaises(TypeError, Instr, 'LOAD_DEREF', 'x')
    Instr('LOAD_DEREF', CellVar('x'))
    Instr('LOAD_DEREF', FreeVar('x'))
    self.assertRaises(TypeError, Instr, 'LOAD_FAST', 1)
    Instr('LOAD_FAST', 'x')
    self.assertRaises(TypeError, Instr, 'LOAD_NAME', 1)
    Instr('LOAD_NAME', 'x')
    self.assertRaises(ValueError, Instr, 'LOAD_CONST')
    self.assertRaises(ValueError, Instr, 'LOAD_CONST', label)
    self.assertRaises(ValueError, Instr, 'LOAD_CONST', block)
    Instr('LOAD_CONST', 1.0)
    Instr('LOAD_CONST', object())
    self.assertRaises(TypeError, Instr, 'COMPARE_OP', 1)
    Instr('COMPARE_OP', Compare.EQ)
    self.assertRaises(ValueError, Instr, 'CALL_FUNCTION', -1)
    self.assertRaises(TypeError, Instr, 'CALL_FUNCTION', 3.0)
    Instr('CALL_FUNCTION', 3)
    self.assertRaises(ValueError, Instr, 'CALL_FUNCTION', 2147483647 + 1)
    instr = Instr('CALL_FUNCTION', 2147483647)
    self.assertEqual(instr.arg, 2147483647)
    self.assertRaises(ValueError, Instr, 'NOP', 0)
    Instr('NOP')