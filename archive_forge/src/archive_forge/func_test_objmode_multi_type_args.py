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
def test_objmode_multi_type_args(self):
    array_ty = types.int32[:]

    @njit
    def foo():
        with objmode(t1='float64', t2=gv_type, t3=array_ty):
            t1 = 793856.5
            t2 = t1
            t3 = np.arange(5).astype(np.int32)
        return (t1, t2, t3)
    t1, t2, t3 = foo()
    self.assertPreciseEqual(t1, 793856.5)
    self.assertPreciseEqual(t2, 793856)
    self.assertPreciseEqual(t3, np.arange(5).astype(np.int32))