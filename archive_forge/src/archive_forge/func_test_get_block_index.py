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
def test_get_block_index(self):
    blocks = ControlFlowGraph()
    block0 = blocks[0]
    block1 = blocks.add_block()
    block2 = blocks.add_block()
    self.assertEqual(blocks.get_block_index(block0), 0)
    self.assertEqual(blocks.get_block_index(block1), 1)
    self.assertEqual(blocks.get_block_index(block2), 2)
    other_block = BasicBlock()
    self.assertRaises(ValueError, blocks.get_block_index, other_block)