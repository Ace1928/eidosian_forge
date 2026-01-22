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
def test_omitted_type(self):

    def inner(a):
        pass

    @overload(inner)
    def inner_overload(a):
        if not isinstance(a, types.Literal):
            return
        return lambda a: a

    @njit
    def my_func(a='a'):
        return inner(a)

    @njit
    def f():
        return my_func()

    @njit
    def g():
        return my_func('b')
    self.assertEqual(f(), 'a')
    self.assertEqual(g(), 'b')