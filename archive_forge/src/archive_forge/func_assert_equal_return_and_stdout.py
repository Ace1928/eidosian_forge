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
def assert_equal_return_and_stdout(self, pyfunc, *args):
    py_args = copy.deepcopy(args)
    c_args = copy.deepcopy(args)
    cfunc = njit(pyfunc)
    with captured_stdout() as stream:
        expect_res = pyfunc(*py_args)
        expect_out = stream.getvalue()
    cfunc.compile(tuple(map(typeof, c_args)))
    with captured_stdout() as stream:
        got_res = cfunc(*c_args)
        got_out = stream.getvalue()
    self.assertEqual(expect_out, got_out)
    self.assertPreciseEqual(expect_res, got_res)