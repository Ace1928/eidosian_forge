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
def test_hash_eq_ne(self):

    class HashEqTest:
        x: int

        def __init__(self, x):
            self.x = x

        def __hash__(self):
            return self.x % 10

        def __eq__(self, o):
            return (self.x - o.x) % 20 == 0

    class HashEqNeTest(HashEqTest):

        def __ne__(self, o):
            return (self.x - o.x) % 20 > 1

    def py_hash(x):
        return hash(x)

    def py_eq(x, y):
        return x == y

    def py_ne(x, y):
        return x != y

    def identity_decorator(f):
        return f
    comparisons = [(0, 1), (2, 22), (7, 10), (3, 3)]
    for base_cls, use_jit in itertools.product([HashEqTest, HashEqNeTest], [False, True]):
        decorator = njit if use_jit else identity_decorator
        hash_func = decorator(py_hash)
        eq_func = decorator(py_eq)
        ne_func = decorator(py_ne)
        jit_cls = jitclass(base_cls)
        for v in [0, 2, 10, 24, -8]:
            self.assertEqual(hash_func(jit_cls(v)), v % 10)
        for x, y in comparisons:
            self.assertEqual(eq_func(jit_cls(x), jit_cls(y)), base_cls(x) == base_cls(y))
            self.assertEqual(ne_func(jit_cls(x), jit_cls(y)), base_cls(x) != base_cls(y))