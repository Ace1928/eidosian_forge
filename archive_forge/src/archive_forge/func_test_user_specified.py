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
def test_user_specified(self):
    rec_dt = np.dtype([('a', np.int32), ('b', np.float32)])
    rec_type = typeof(rec_dt)

    @jit((rec_type[:],), nopython=True)
    def foo(x):
        return (x['a'], x['b'])
    arr = np.zeros(1, dtype=rec_dt)
    arr[0]['a'] = 123
    arr[0]['b'] = 32.1
    a, b = foo(arr)
    self.assertEqual(a, arr[0]['a'])
    self.assertEqual(b, arr[0]['b'])