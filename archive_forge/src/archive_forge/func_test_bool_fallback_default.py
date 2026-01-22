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
def test_bool_fallback_default(self):

    class NoBoolNoLen:

        def __init__(self):
            pass

        def get_bool(self):
            return bool(self)
    py_class = NoBoolNoLen
    jitted_class = jitclass([])(py_class)
    py_class_bool = py_class().get_bool()
    jitted_class_bool = jitted_class().get_bool()
    self.assertEqual(py_class_bool, jitted_class_bool)
    self.assertEqual(type(py_class_bool), type(jitted_class_bool))