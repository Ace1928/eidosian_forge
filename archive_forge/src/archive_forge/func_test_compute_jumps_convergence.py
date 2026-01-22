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
def test_compute_jumps_convergence(self):
    code = Bytecode()
    label1 = Label()
    label2 = Label()
    nop = 'NOP'
    code.append(Instr('JUMP_ABSOLUTE', label1))
    code.append(Instr('JUMP_ABSOLUTE', label2))
    for x in range(4, 510 if OFFSET_AS_INSTRUCTION else 254, 2):
        code.append(Instr(nop))
    code.append(label1)
    code.append(Instr(nop))
    for x in range(514 if OFFSET_AS_INSTRUCTION else 256, 600 if OFFSET_AS_INSTRUCTION else 300, 2):
        code.append(Instr(nop))
    code.append(label2)
    code.append(Instr(nop))
    code.to_code()
    with self.assertRaises(RuntimeError):
        code.to_code(compute_jumps_passes=2)