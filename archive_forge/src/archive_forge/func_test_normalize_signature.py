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
def test_normalize_signature(self):
    f = sigutils.normalize_signature

    def check(sig, args, return_type):
        self.assertEqual(f(sig), (args, return_type))

    def check_error(sig, msg):
        with self.assertRaises(TypeError) as raises:
            f(sig)
        self.assertIn(msg, str(raises.exception))
    f32 = types.float32
    c64 = types.complex64
    i16 = types.int16
    a = types.Array(f32, 1, 'C')
    check((c64,), (c64,), None)
    check((f32, i16), (f32, i16), None)
    check(a(i16), (i16,), a)
    check('int16(complex64)', (c64,), i16)
    check('(complex64, int16)', (c64, i16), None)
    check(typing.signature(i16, c64), (c64,), i16)
    msg = 'invalid type in signature: expected a type instance'
    check_error((types.Integer,), msg)
    check_error((None,), msg)
    check_error([], 'invalid signature')