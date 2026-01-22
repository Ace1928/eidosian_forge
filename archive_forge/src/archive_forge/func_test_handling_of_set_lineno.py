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
def test_handling_of_set_lineno(self):
    code = Bytecode()
    code.first_lineno = 3
    code.extend([Instr('LOAD_CONST', 7), Instr('STORE_NAME', 'x'), SetLineno(4), Instr('LOAD_CONST', 8), Instr('STORE_NAME', 'y'), SetLineno(5), Instr('LOAD_CONST', 9), Instr('STORE_NAME', 'z')])
    self.assertEqual(code.compute_stacksize(), 1)