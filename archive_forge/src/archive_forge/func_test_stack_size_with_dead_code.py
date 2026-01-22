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
def test_stack_size_with_dead_code(self):

    def test(*args):
        return 0
        try:
            a = args[0]
        except IndexError:
            return -1
        else:
            return a
    test.__code__ = Bytecode.from_code(test.__code__).to_code()
    self.assertEqual(test.__code__.co_stacksize, 1)
    self.assertEqual(test(1), 0)