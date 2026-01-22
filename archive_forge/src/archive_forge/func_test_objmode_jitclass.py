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
def test_objmode_jitclass(self):
    spec = [('value', types.int32), ('array', types.float32[:])]

    @jitclass(spec)
    class Bag(object):

        def __init__(self, value):
            self.value = value
            self.array = np.zeros(value, dtype=np.float32)

        @property
        def size(self):
            return self.array.size

        def increment(self, val):
            for i in range(self.size):
                self.array[i] += val
            return self.array

        @staticmethod
        def add(x, y):
            return x + y
    n = 21
    mybag = Bag(n)

    def foo():
        pass

    @overload(foo)
    def foo_overload():
        shrubbery = mybag._numba_type_

        def impl():
            with objmode(out=shrubbery):
                out = Bag(123)
                out.increment(3)
            return out
        return impl

    @njit
    def bar():
        return foo()
    z = bar()
    self.assertIsInstance(z, Bag)
    self.assertEqual(z.add(2, 3), 2 + 3)
    exp_array = np.zeros(123, dtype=np.float32) + 3
    self.assertPreciseEqual(z.array, exp_array)