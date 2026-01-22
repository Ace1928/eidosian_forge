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
def test_jitclass_decorator_usecases(self):
    spec = OrderedDict(x=float64)

    @jitclass()
    class Test1:
        x: float

        def __init__(self):
            self.x = 0
    self.assertIsInstance(Test1, JitClassType)
    self.assertDictEqual(Test1.class_type.struct, spec)

    @jitclass(spec=spec)
    class Test2:

        def __init__(self):
            self.x = 0
    self.assertIsInstance(Test2, JitClassType)
    self.assertDictEqual(Test2.class_type.struct, spec)

    @jitclass
    class Test3:
        x: float

        def __init__(self):
            self.x = 0
    self.assertIsInstance(Test3, JitClassType)
    self.assertDictEqual(Test3.class_type.struct, spec)

    @jitclass(spec)
    class Test4:

        def __init__(self):
            self.x = 0
    self.assertIsInstance(Test4, JitClassType)
    self.assertDictEqual(Test4.class_type.struct, spec)