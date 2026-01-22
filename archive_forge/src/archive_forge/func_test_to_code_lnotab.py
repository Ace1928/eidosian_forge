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
def test_to_code_lnotab(self):

    def f():
        x = 7
        y = 8
        z = 9
    fl = f.__code__.co_firstlineno
    concrete = ConcreteBytecode()
    concrete.consts = [None, 7, 8, 9]
    concrete.varnames = ['x', 'y', 'z']
    concrete.first_lineno = fl
    concrete.extend([SetLineno(fl + 3), ConcreteInstr('LOAD_CONST', 1), ConcreteInstr('STORE_FAST', 0), SetLineno(fl + 4), ConcreteInstr('LOAD_CONST', 2), ConcreteInstr('STORE_FAST', 1), SetLineno(fl + 5), ConcreteInstr('LOAD_CONST', 3), ConcreteInstr('STORE_FAST', 2), ConcreteInstr('LOAD_CONST', 0), ConcreteInstr('RETURN_VALUE')])
    code = concrete.to_code()
    self.assertEqual(code.co_code, f.__code__.co_code)
    self.assertEqual(code.co_lnotab, f.__code__.co_lnotab)
    if sys.version_info >= (3, 10):
        self.assertEqual(code.co_linetable, f.__code__.co_linetable)