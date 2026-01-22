import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
def test_structref_type(self):
    sr = types.StructRef([('a', types.int64)])
    self.assertEqual(sr.field_dict['a'], types.int64)
    sr = types.StructRef([('a', types.int64), ('b', types.float64)])
    self.assertEqual(sr.field_dict['a'], types.int64)
    self.assertEqual(sr.field_dict['b'], types.float64)
    with self.assertRaisesRegex(ValueError, 'expecting a str for field name'):
        types.StructRef([(1, types.int64)])
    with self.assertRaisesRegex(ValueError, 'expecting a Numba Type for field type'):
        types.StructRef([('a', 123)])