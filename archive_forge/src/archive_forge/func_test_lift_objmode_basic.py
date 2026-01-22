import copy
import warnings
import numpy as np
import numba
from numba.core.transforms import find_setupwiths, with_lifting
from numba.core.withcontexts import bypass_context, call_context, objmode_context
from numba.core.bytecode import FunctionIdentity, ByteCode
from numba.core.interpreter import Interpreter
from numba.core import errors
from numba.core.registry import cpu_target
from numba.core.compiler import compile_ir, DEFAULT_FLAGS
from numba import njit, typeof, objmode, types
from numba.core.extending import overload
from numba.tests.support import (MemoryLeak, TestCase, captured_stdout,
from numba.core.utils import PYVERSION
from numba.experimental import jitclass
import unittest
def test_lift_objmode_basic(self):

    def bar(ival):
        print('ival =', {'ival': ival // 2})

    def foo(ival):
        ival += 1
        with objmode_context:
            bar(ival)
        return ival + 1

    def foo_nonglobal(ival):
        ival += 1
        with numba.objmode:
            bar(ival)
        return ival + 1
    self.assert_equal_return_and_stdout(foo, 123)
    self.assert_equal_return_and_stdout(foo_nonglobal, 123)