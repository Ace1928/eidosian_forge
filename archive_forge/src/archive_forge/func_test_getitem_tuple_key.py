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
def test_getitem_tuple_key(self):
    spec = [('data', int32[:, :])]

    @jitclass(spec)
    class TestClass(object):

        def __init__(self):
            self.data = np.zeros((10, 10), dtype=np.int32)

        def __setitem__(self, key, data):
            self.data[key[0], key[1]] = data

        def __getitem__(self, key):
            return self.data[key[0], key[1]]
    t = TestClass()
    t[1, 1] = 11

    @njit
    def get11(t):
        return t[1, 1]

    @njit
    def set22(t, data):
        t[2, 2] = data
    self.assertEqual(get11(t), 11)
    set22(t, 22)
    self.assertEqual(t[2, 2], 22)