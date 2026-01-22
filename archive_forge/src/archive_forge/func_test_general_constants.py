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
def test_general_constants(self):
    """Test if general object could be linked as constants."""

    class CustomObject:
        pass

    class UnHashableCustomObject:
        __hash__ = None
    obj1 = [1, 2, 3]
    obj2 = {1, 2, 3}
    obj3 = CustomObject()
    obj4 = UnHashableCustomObject()
    code = Bytecode([Instr('LOAD_CONST', obj1, lineno=1), Instr('LOAD_CONST', obj2, lineno=1), Instr('LOAD_CONST', obj3, lineno=1), Instr('LOAD_CONST', obj4, lineno=1), Instr('BUILD_TUPLE', 4, lineno=1), Instr('RETURN_VALUE', lineno=1)])
    self.assertEqual(code.to_code().co_consts, (obj1, obj2, obj3, obj4))

    def f():
        return
    f.__code__ = code.to_code()
    self.assertEqual(f(), (obj1, obj2, obj3, obj4))