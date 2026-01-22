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
def test_jitclass_usage_from_python(self):
    Float2AndArray = self._make_Float2AndArray()

    @njit
    def identity(obj):
        return obj

    @njit
    def retrieve_attributes(obj):
        return (obj.x, obj.y, obj.arr)
    arr = np.arange(10, dtype=np.float32)
    obj = Float2AndArray(1, 2, arr)
    obj_meminfo = _get_meminfo(obj)
    self.assertEqual(obj_meminfo.refcount, 2)
    self.assertEqual(obj_meminfo.data, _box.box_get_dataptr(obj))
    self.assertEqual(obj._numba_type_.class_type, Float2AndArray.class_type)
    other = identity(obj)
    other_meminfo = _get_meminfo(other)
    self.assertEqual(obj_meminfo.refcount, 4)
    self.assertEqual(other_meminfo.refcount, 4)
    self.assertEqual(other_meminfo.data, _box.box_get_dataptr(other))
    self.assertEqual(other_meminfo.data, obj_meminfo.data)
    del other, other_meminfo
    self.assertEqual(obj_meminfo.refcount, 2)
    out_x, out_y, out_arr = retrieve_attributes(obj)
    self.assertEqual(out_x, 1)
    self.assertEqual(out_y, 2)
    self.assertIs(out_arr, arr)
    self.assertEqual(obj.x, 1)
    self.assertEqual(obj.y, 2)
    self.assertIs(obj.arr, arr)
    self.assertEqual(obj.add(123), 123)
    self.assertEqual(obj.x, 1 + 123)
    self.assertEqual(obj.y, 2 + 123)
    obj.x = 333
    obj.y = 444
    obj.arr = newarr = np.arange(5, dtype=np.float32)
    self.assertEqual(obj.x, 333)
    self.assertEqual(obj.y, 444)
    self.assertIs(obj.arr, newarr)