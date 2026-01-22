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
def test_type_of_literal(self):

    def inner(a):
        pass

    @overload(inner)
    def inner_overload(a):
        if not isinstance(a, types.Literal):
            return
        self.assertIsInstance(a, types.Literal)
        return lambda a: type(a)(a + 1.23)

    @njit
    def my_func(a=1):
        return inner(a)

    @njit
    def f():
        return my_func()

    @njit
    def g():
        return my_func(100)
    self.assertEqual(f(), 2)
    self.assertEqual(g(), 101)