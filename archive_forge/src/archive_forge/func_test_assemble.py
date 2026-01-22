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
def test_assemble(self):
    instr = ConcreteInstr('NOP')
    self.assertEqual(instr.assemble(), b'\t\x00')
    instr = ConcreteInstr('LOAD_CONST', 3)
    self.assertEqual(instr.assemble(), b'd\x03')
    instr = ConcreteInstr('LOAD_CONST', 305441741)
    self.assertEqual(instr.assemble(), b'\x90\x12\x904\x90\xabd\xcd')
    instr = ConcreteInstr('LOAD_CONST', 3, extended_args=1)
    self.assertEqual(instr.assemble(), b'\x90\x00d\x03')