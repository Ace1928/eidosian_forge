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
def test_label2(self):
    bytecode = Bytecode()
    label = Label()
    bytecode.extend([Instr('LOAD_NAME', 'test', lineno=1), Instr('POP_JUMP_IF_FALSE', label), Instr('LOAD_CONST', 5, lineno=2), Instr('STORE_NAME', 'x'), Instr('JUMP_FORWARD', label), Instr('LOAD_CONST', 7, lineno=4), Instr('STORE_NAME', 'x'), label, Instr('LOAD_CONST', None), Instr('RETURN_VALUE')])
    concrete = bytecode.to_concrete_bytecode()
    expected = [ConcreteInstr('LOAD_NAME', 0, lineno=1), ConcreteInstr('POP_JUMP_IF_FALSE', 7 if OFFSET_AS_INSTRUCTION else 14, lineno=1), ConcreteInstr('LOAD_CONST', 0, lineno=2), ConcreteInstr('STORE_NAME', 1, lineno=2), ConcreteInstr('JUMP_FORWARD', 2 if OFFSET_AS_INSTRUCTION else 4, lineno=2), ConcreteInstr('LOAD_CONST', 1, lineno=4), ConcreteInstr('STORE_NAME', 1, lineno=4), ConcreteInstr('LOAD_CONST', 2, lineno=4), ConcreteInstr('RETURN_VALUE', lineno=4)]
    self.assertListEqual(list(concrete), expected)
    self.assertListEqual(concrete.consts, [5, 7, None])
    self.assertListEqual(concrete.names, ['test', 'x'])
    self.assertListEqual(concrete.varnames, [])