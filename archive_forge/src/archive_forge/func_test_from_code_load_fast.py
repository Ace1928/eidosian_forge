import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import textwrap
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, FreeVar, Bytecode, SetLineno, ConcreteInstr
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, get_code
def test_from_code_load_fast(self):
    code = get_code('\n            def func():\n                x = 33\n                y = x\n        ', function=True)
    code = Bytecode.from_code(code)
    self.assertEqual(code, [Instr('LOAD_CONST', 33, lineno=2), Instr('STORE_FAST', 'x', lineno=2), Instr('LOAD_FAST', 'x', lineno=3), Instr('STORE_FAST', 'y', lineno=3), Instr('LOAD_CONST', None, lineno=3), Instr('RETURN_VALUE', lineno=3)])