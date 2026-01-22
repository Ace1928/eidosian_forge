import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.flags import infer_flags
def test_async_gen_no_flag_is_async_True(self):
    code = ConcreteBytecode()
    code.update_flags(is_async=True)
    self.assertTrue(bool(code.flags & CompilerFlags.COROUTINE))
    for i, expected in (('YIELD_VALUE', CompilerFlags.ASYNC_GENERATOR), ('YIELD_FROM', CompilerFlags.COROUTINE)):
        code = ConcreteBytecode()
        code.append(ConcreteInstr(i))
        code.update_flags(is_async=True)
        self.assertTrue(bool(code.flags & expected))