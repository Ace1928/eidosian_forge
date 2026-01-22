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
def test_name_shadowing_error(self):

    class Foo(object):

        def __init__(self):
            pass

        @property
        def my_property(self):
            pass

        def my_method(self):
            pass
    with self.assertRaises(NameError) as raises:
        jitclass(Foo, [('my_property', int32)])
    self.assertEqual(str(raises.exception), 'name shadowing: my_property')
    with self.assertRaises(NameError) as raises:
        jitclass(Foo, [('my_method', int32)])
    self.assertEqual(str(raises.exception), 'name shadowing: my_method')