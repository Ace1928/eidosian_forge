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
def test_function_incompatible_templates(self):

    def func_stub():
        pass

    def func_stub2():
        pass

    def ol():
        pass
    template1 = make_overload_template(func_stub, ol, {}, True, 'never')
    template2 = make_overload_template(func_stub2, ol, {}, True, 'never')
    with self.assertRaises(ValueError) as raises:
        types.Function((template1, template2))
    self.assertIn('incompatible templates:', str(raises.exception))