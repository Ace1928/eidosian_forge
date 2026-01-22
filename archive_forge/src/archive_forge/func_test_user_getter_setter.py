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
def test_user_getter_setter(self):

    @jitclass([('attr', int32)])
    class Foo(object):

        def __init__(self, attr):
            self.attr = attr

        @property
        def value(self):
            return self.attr + 1

        @value.setter
        def value(self, val):
            self.attr = val - 1
    foo = Foo(123)
    self.assertEqual(foo.attr, 123)
    self.assertEqual(foo.value, 123 + 1)
    foo.value = 789
    self.assertEqual(foo.attr, 789 - 1)
    self.assertEqual(foo.value, 789)

    @njit
    def bar(foo, val):
        a = foo.value
        foo.value = val
        b = foo.value
        c = foo.attr
        return (a, b, c)
    a, b, c = bar(foo, 567)
    self.assertEqual(a, 789)
    self.assertEqual(b, 567)
    self.assertEqual(c, 567 - 1)