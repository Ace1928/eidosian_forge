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
@expected_failure_for_list_arg
def test_case10_mutate_across_contexts(self):

    def foo(x):
        with objmode_context(y='List(int64)'):
            y = [1, 2, 3]
        with objmode_context():
            y[2] = 10
        return y
    x = np.array([1, 2, 3])
    self.assert_equal_return_and_stdout(foo, x)