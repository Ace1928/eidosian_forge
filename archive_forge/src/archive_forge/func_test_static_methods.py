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
def test_static_methods(self):

    @jitclass([('x', int32)])
    class Test1:

        def __init__(self, x):
            self.x = x

        def increase(self, y):
            self.x = self.add(self.x, y)
            return self.x

        @staticmethod
        def add(a, b):
            return a + b

        @staticmethod
        def sub(a, b):
            return a - b

    @jitclass([('x', int32)])
    class Test2:

        def __init__(self, x):
            self.x = x

        def increase(self, y):
            self.x = self.add(self.x, y)
            return self.x

        @staticmethod
        def add(a, b):
            return a - b
    self.assertIsInstance(Test1.add, Dispatcher)
    self.assertIsInstance(Test1.sub, Dispatcher)
    self.assertIsInstance(Test2.add, Dispatcher)
    self.assertNotEqual(Test1.add, Test2.add)
    self.assertEqual(3, Test1.add(1, 2))
    self.assertEqual(-1, Test2.add(1, 2))
    self.assertEqual(4, Test1.sub(6, 2))
    t1 = Test1(0)
    t2 = Test2(0)
    self.assertEqual(1, t1.increase(1))
    self.assertEqual(-1, t2.increase(1))
    self.assertEqual(2, t1.add(1, 1))
    self.assertEqual(0, t1.sub(1, 1))
    self.assertEqual(0, t2.add(1, 1))
    self.assertEqual(2j, t1.add(1j, 1j))
    self.assertEqual(1j, t1.sub(2j, 1j))
    self.assertEqual('foobar', t1.add('foo', 'bar'))
    with self.assertRaises(AttributeError) as raises:
        Test2.sub(3, 1)
    self.assertIn("has no attribute 'sub'", str(raises.exception))
    with self.assertRaises(TypeError) as raises:
        Test1.add(3)
    self.assertIn('not enough arguments: expected 2, got 1', str(raises.exception))

    @jitclass([])
    class Test3:

        def __init__(self):
            pass

        @staticmethod
        def a_static_method(a, b):
            pass

        def call_static(self):
            return Test3.a_static_method(1, 2)
    invalid = Test3()
    with self.assertRaises(errors.TypingError) as raises:
        invalid.call_static()
    self.assertIn("Unknown attribute 'a_static_method'", str(raises.exception))