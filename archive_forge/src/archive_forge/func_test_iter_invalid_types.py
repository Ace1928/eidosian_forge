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
def test_iter_invalid_types(self):
    block = BasicBlock()
    block.append(Label())
    with self.assertRaises(ValueError):
        list(block)
    with self.assertRaises(ValueError):
        block.legalize(1)
    block = BasicBlock()
    block2 = BasicBlock()
    block.extend([Instr('JUMP_ABSOLUTE', block2), Instr('NOP')])
    with self.assertRaises(ValueError):
        list(block)
    with self.assertRaises(ValueError):
        block.legalize(1)
    block = BasicBlock()
    label = Label()
    block.extend([Instr('JUMP_ABSOLUTE', label)])
    with self.assertRaises(ValueError):
        list(block)
    with self.assertRaises(ValueError):
        block.legalize(1)