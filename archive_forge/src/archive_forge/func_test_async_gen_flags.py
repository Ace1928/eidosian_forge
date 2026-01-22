import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.flags import infer_flags
def test_async_gen_flags(self):
    for is_async in (None, True):
        code = ConcreteBytecode()
        code.append(ConcreteInstr('YIELD_VALUE'))
        for f, expected in ((CompilerFlags.COROUTINE, CompilerFlags.ASYNC_GENERATOR), (CompilerFlags.ASYNC_GENERATOR, CompilerFlags.ASYNC_GENERATOR), (CompilerFlags.ITERABLE_COROUTINE, CompilerFlags.ITERABLE_COROUTINE)):
            code.flags = CompilerFlags(f)
            code.update_flags(is_async=is_async)
            self.assertTrue(bool(code.flags & expected))
        code = ConcreteBytecode()
        code.append(ConcreteInstr('YIELD_FROM'))
        for f, expected in ((CompilerFlags.COROUTINE, CompilerFlags.COROUTINE), (CompilerFlags.ASYNC_GENERATOR, CompilerFlags.COROUTINE), (CompilerFlags.ITERABLE_COROUTINE, CompilerFlags.ITERABLE_COROUTINE)):
            code.flags = CompilerFlags(f)
            code.update_flags(is_async=is_async)
            self.assertTrue(bool(code.flags & expected))
        code = ConcreteBytecode()
        code.append(ConcreteInstr('GET_AWAITABLE'))
        code.flags = CompilerFlags(CompilerFlags.ITERABLE_COROUTINE)
        with self.assertRaises(ValueError):
            code.update_flags(is_async=is_async)