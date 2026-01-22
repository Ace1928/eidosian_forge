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
def check_stack_size(self, func):
    code = func.__code__
    bytecode = Bytecode.from_code(code)
    cfg = ControlFlowGraph.from_bytecode(bytecode)
    self.assertEqual(code.co_stacksize, cfg.compute_stacksize())