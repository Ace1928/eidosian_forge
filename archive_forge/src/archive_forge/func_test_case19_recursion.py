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
@expected_failure_py311
@expected_failure_py312
def test_case19_recursion(self):

    def foo(x):
        with objmode_context():
            if x == 0:
                return 7
        ret = foo(x - 1)
        return ret
    with self.assertRaises((errors.TypingError, errors.CompilerError)) as raises:
        cfoo = njit(foo)
        cfoo(np.array([1, 2, 3]))
    msg = "Untyped global name 'foo'"
    self.assertIn(msg, str(raises.exception))