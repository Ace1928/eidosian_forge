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
def test_issue_typeref_key(self):

    class NoUniqueNameType(types.Dummy):

        def __init__(self, param):
            super(NoUniqueNameType, self).__init__('NoUniqueNameType')
            self.param = param

        @property
        def key(self):
            return self.param
    no_unique_name_type_1 = NoUniqueNameType(1)
    no_unique_name_type_2 = NoUniqueNameType(2)
    for ty1 in (no_unique_name_type_1, no_unique_name_type_2):
        for ty2 in (no_unique_name_type_1, no_unique_name_type_2):
            self.assertIs(types.TypeRef(ty1) == types.TypeRef(ty2), ty1 == ty2)