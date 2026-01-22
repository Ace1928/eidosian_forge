import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import opcode
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase
def test_is_cond_jump(self):
    label = Label()
    jump = Instr('POP_JUMP_IF_TRUE', label)
    self.assertTrue(jump.is_cond_jump())
    instr = Instr('LOAD_FAST', 'x')
    self.assertFalse(instr.is_cond_jump())