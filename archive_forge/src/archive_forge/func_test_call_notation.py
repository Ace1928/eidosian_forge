from collections import namedtuple
import gc
import os
import operator
import sys
import weakref
import numpy as np
from numba.core import types, typing, errors, sigutils
from numba.core.types.abstract import _typecache
from numba.core.types.functions import _header_lead
from numba.core.typing.templates import make_overload_template
from numba import jit, njit, typeof
from numba.core.extending import (overload, register_model, models, unbox,
from numba.tests.support import TestCase, create_temp_module
from numba.tests.enum_usecases import Color, Shake, Shape
import unittest
from numba.np import numpy_support
from numba.core import types
def test_call_notation(self):
    i = types.int32
    d = types.double
    self.assertEqual(i(), typing.signature(i))
    self.assertEqual(i(d), typing.signature(i, d))
    self.assertEqual(i(d, d), typing.signature(i, d, d))
    self.assertPreciseEqual(i(42.5), 42)
    self.assertPreciseEqual(d(-5), -5.0)
    ty = types.NPDatetime('Y')
    self.assertPreciseEqual(ty('1900'), np.datetime64('1900', 'Y'))
    self.assertPreciseEqual(ty('NaT'), np.datetime64('NaT', 'Y'))
    ty = types.NPTimedelta('s')
    self.assertPreciseEqual(ty(5), np.timedelta64(5, 's'))
    self.assertPreciseEqual(ty('NaT'), np.timedelta64('NaT', 's'))
    ty = types.NPTimedelta('')
    self.assertPreciseEqual(ty(5), np.timedelta64(5))
    self.assertPreciseEqual(ty('NaT'), np.timedelta64('NaT'))