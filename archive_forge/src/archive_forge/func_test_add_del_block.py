import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import io
import sys
import unittest
import contextlib
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.concrete import OFFSET_AS_INSTRUCTION
from _pydevd_frame_eval.vendored.bytecode.tests import disassemble as _disassemble, TestCase
def test_add_del_block(self):
    code = ControlFlowGraph()
    code[0].append(Instr('LOAD_CONST', 0))
    block = code.add_block()
    self.assertEqual(len(code), 2)
    self.assertIs(block, code[1])
    code[1].append(Instr('LOAD_CONST', 2))
    self.assertBlocksEqual(code, [Instr('LOAD_CONST', 0)], [Instr('LOAD_CONST', 2)])
    del code[0]
    self.assertBlocksEqual(code, [Instr('LOAD_CONST', 2)])
    del code[0]
    self.assertEqual(len(code), 0)