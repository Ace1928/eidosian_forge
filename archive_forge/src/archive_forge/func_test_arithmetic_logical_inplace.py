import ctypes
import itertools
import pickle
import random
import typing as pt
import unittest
from collections import OrderedDict
import numpy as np
from numba import (boolean, deferred_type, float32, float64, int16, int32,
from numba.core import errors, types
from numba.core.dispatcher import Dispatcher
from numba.core.errors import LoweringError, TypingError
from numba.core.runtime.nrt import MemInfo
from numba.experimental import jitclass
from numba.experimental.jitclass import _box
from numba.experimental.jitclass.base import JitClassType
from numba.tests.support import MemoryLeakMixin, TestCase, skip_if_typeguard
from numba.tests.support import skip_unless_scipy
def test_arithmetic_logical_inplace(self):
    JitIntWrapper = self.get_int_wrapper()
    JitFloatWrapper = self.get_float_wrapper()
    PyIntWrapper = JitIntWrapper.mro()[1]
    PyFloatWrapper = JitFloatWrapper.mro()[1]

    @jitclass([('x', types.intp)])
    class JitIntUpdateWrapper(PyIntWrapper):

        def __init__(self, value):
            self.x = value

        def __ilshift__(self, other):
            return JitIntUpdateWrapper(self.x << other.x)

        def __irshift__(self, other):
            return JitIntUpdateWrapper(self.x >> other.x)

        def __iand__(self, other):
            return JitIntUpdateWrapper(self.x & other.x)

        def __ior__(self, other):
            return JitIntUpdateWrapper(self.x | other.x)

        def __ixor__(self, other):
            return JitIntUpdateWrapper(self.x ^ other.x)

    @jitclass({'x': types.float64})
    class JitFloatUpdateWrapper(PyFloatWrapper):

        def __init__(self, value):
            self.x = value

        def __iadd__(self, other):
            return JitFloatUpdateWrapper(self.x + 2.718 * other.x)

        def __ifloordiv__(self, other):
            return JitFloatUpdateWrapper(self.x * 2.718 // other.x)

        def __imod__(self, other):
            return JitFloatUpdateWrapper(self.x % (other.x + 1))

        def __imul__(self, other):
            return JitFloatUpdateWrapper(self.x * other.x + 1)

        def __ipow__(self, other):
            return JitFloatUpdateWrapper(self.x ** other.x + 1)

        def __isub__(self, other):
            return JitFloatUpdateWrapper(self.x - 3.1415 * other.x)

        def __itruediv__(self, other):
            return JitFloatUpdateWrapper((self.x + 1) / other.x)
    PyIntUpdateWrapper = JitIntUpdateWrapper.mro()[1]
    PyFloatUpdateWrapper = JitFloatUpdateWrapper.mro()[1]

    def get_update_func(op):
        template = f'\ndef f(x, y):\n    x {op}= y\n    return x\n'
        namespace = {}
        exec(template, namespace)
        return namespace['f']
    float_py_funcs = [get_update_func(op) for op in ['+', '//', '%', '*', '**', '-', '/']]
    int_py_funcs = [get_update_func(op) for op in ['<<', '>>', '&', '|', '^']]
    test_values = [(0.0, 2.0), (1.234, 3.1415), (13.1, 1.01)]
    for jit_f, (py_cls, jit_cls), (x, y) in itertools.product(map(njit, float_py_funcs), [(PyFloatWrapper, JitFloatWrapper), (PyFloatUpdateWrapper, JitFloatUpdateWrapper)], test_values):
        py_f = jit_f.py_func
        expected = py_f(py_cls(x), py_cls(y)).x
        self.assertAlmostEqual(expected, py_f(jit_cls(x), jit_cls(y)).x)
        self.assertAlmostEqual(expected, jit_f(jit_cls(x), jit_cls(y)).x)
    for jit_f, (py_cls, jit_cls), (x, y) in itertools.product(map(njit, int_py_funcs), [(PyIntWrapper, JitIntWrapper), (PyIntUpdateWrapper, JitIntUpdateWrapper)], test_values):
        x, y = (int(x), int(y))
        py_f = jit_f.py_func
        expected = py_f(py_cls(x), py_cls(y)).x
        self.assertEqual(expected, py_f(jit_cls(x), jit_cls(y)).x)
        self.assertEqual(expected, jit_f(jit_cls(x), jit_cls(y)).x)