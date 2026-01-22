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
def test_explicit_stacksize(self):
    code_obj = get_code("print('%s' % (a,b,c))")
    original_stacksize = code_obj.co_stacksize
    concrete = ConcreteBytecode.from_code(code_obj)
    explicit_stacksize = original_stacksize + 42
    new_code_obj = concrete.to_code(stacksize=explicit_stacksize)
    self.assertEqual(new_code_obj.co_stacksize, explicit_stacksize)
    explicit_stacksize = 0
    new_code_obj = concrete.to_code(stacksize=explicit_stacksize)
    self.assertEqual(new_code_obj.co_stacksize, explicit_stacksize)