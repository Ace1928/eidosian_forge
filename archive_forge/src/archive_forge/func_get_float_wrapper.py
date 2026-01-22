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
@staticmethod
def get_float_wrapper():

    @jitclass([('x', types.float64)])
    class FloatWrapper:

        def __init__(self, value):
            self.x = value

        def __eq__(self, other):
            return self.x == other.x

        def __hash__(self):
            return self.x

        def __ge__(self, other):
            return self.x >= other.x

        def __gt__(self, other):
            return self.x > other.x

        def __le__(self, other):
            return self.x <= other.x

        def __lt__(self, other):
            return self.x < other.x

        def __add__(self, other):
            return FloatWrapper(self.x + other.x)

        def __floordiv__(self, other):
            return FloatWrapper(self.x // other.x)

        def __mod__(self, other):
            return FloatWrapper(self.x % other.x)

        def __mul__(self, other):
            return FloatWrapper(self.x * other.x)

        def __neg__(self, other):
            return FloatWrapper(-self.x)

        def __pos__(self, other):
            return FloatWrapper(+self.x)

        def __pow__(self, other):
            return FloatWrapper(self.x ** other.x)

        def __sub__(self, other):
            return FloatWrapper(self.x - other.x)

        def __truediv__(self, other):
            return FloatWrapper(self.x / other.x)
    return FloatWrapper