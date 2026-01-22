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
def test_operator_reflection(self):

    class OperatorsDefined:

        def __init__(self, x):
            self.x = x

        def __eq__(self, other):
            return self.x == other.x

        def __le__(self, other):
            return self.x <= other.x

        def __lt__(self, other):
            return self.x < other.x

        def __ge__(self, other):
            return self.x >= other.x

        def __gt__(self, other):
            return self.x > other.x

    class NoOperatorsDefined:

        def __init__(self, x):
            self.x = x
    spec = [('x', types.int32)]
    JitOperatorsDefined = jitclass(spec)(OperatorsDefined)
    JitNoOperatorsDefined = jitclass(spec)(NoOperatorsDefined)
    py_ops_defined = OperatorsDefined(2)
    py_ops_not_defined = NoOperatorsDefined(3)
    jit_ops_defined = JitOperatorsDefined(2)
    jit_ops_not_defined = JitNoOperatorsDefined(3)
    self.assertEqual(py_ops_not_defined == py_ops_defined, jit_ops_not_defined == jit_ops_defined)
    self.assertEqual(py_ops_not_defined <= py_ops_defined, jit_ops_not_defined <= jit_ops_defined)
    self.assertEqual(py_ops_not_defined < py_ops_defined, jit_ops_not_defined < jit_ops_defined)
    self.assertEqual(py_ops_not_defined >= py_ops_defined, jit_ops_not_defined >= jit_ops_defined)
    self.assertEqual(py_ops_not_defined > py_ops_defined, jit_ops_not_defined > jit_ops_defined)