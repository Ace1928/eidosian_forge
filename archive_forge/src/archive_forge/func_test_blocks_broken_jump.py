import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import contextlib
import io
import sys
import textwrap
import unittest
from _pydevd_frame_eval.vendored import bytecode
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Bytecode, BasicBlock, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode.concrete import OFFSET_AS_INSTRUCTION
from _pydevd_frame_eval.vendored.bytecode.tests import disassemble
def test_blocks_broken_jump(self):
    block = BasicBlock()
    code = ControlFlowGraph()
    code[0].append(Instr('JUMP_ABSOLUTE', block))
    expected = textwrap.dedent('\n            block1:\n                JUMP_ABSOLUTE <error: unknown block>\n\n        ').lstrip('\n')
    self.check_dump_bytecode(code, expected)