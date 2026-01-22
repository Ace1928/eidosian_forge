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
def test_protected_attrs(self):
    spec = {'value': int32, '_value': float32, '__value': int32, '__value__': int32}

    @jitclass(spec)
    class MyClass(object):

        def __init__(self, value):
            self.value = value
            self._value = value / 2
            self.__value = value * 2
            self.__value__ = value - 1

        @property
        def private_value(self):
            return self.__value

        @property
        def _inner_value(self):
            return self._value

        @_inner_value.setter
        def _inner_value(self, v):
            self._value = v

        @property
        def __private_value(self):
            return self.__value

        @__private_value.setter
        def __private_value(self, v):
            self.__value = v

        def swap_private_value(self, new):
            old = self.__private_value
            self.__private_value = new
            return old

        def _protected_method(self, factor):
            return self._value * factor

        def __private_method(self, factor):
            return self.__value * factor

        def check_private_method(self, factor):
            return self.__private_method(factor)
    value = 123
    inst = MyClass(value)
    self.assertEqual(inst.value, value)
    self.assertEqual(inst._value, value / 2)
    self.assertEqual(inst.private_value, value * 2)
    self.assertEqual(inst._inner_value, inst._value)
    freeze_inst_value = inst._value
    inst._inner_value -= 1
    self.assertEqual(inst._inner_value, freeze_inst_value - 1)
    self.assertEqual(inst.swap_private_value(321), value * 2)
    self.assertEqual(inst.swap_private_value(value * 2), 321)
    self.assertEqual(inst._protected_method(3), inst._value * 3)
    self.assertEqual(inst.check_private_method(3), inst.private_value * 3)
    self.assertEqual(inst.__value__, value - 1)
    inst.__value__ -= 100
    self.assertEqual(inst.__value__, value - 101)

    @njit
    def access_dunder(inst):
        return inst.__value
    with self.assertRaises(errors.TypingError) as raises:
        access_dunder(inst)
    self.assertIn('_TestJitClass__value', str(raises.exception))
    with self.assertRaises(AttributeError) as raises:
        access_dunder.py_func(inst)
    self.assertIn('_TestJitClass__value', str(raises.exception))