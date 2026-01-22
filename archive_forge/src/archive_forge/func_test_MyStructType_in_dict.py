import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
def test_MyStructType_in_dict(self):
    td = typed.Dict()
    td['a'] = MyStruct(1, 2.3)
    self.assertEqual(td['a'].values, 1)
    self.assertEqual(td['a'].counter, 2.3)
    td['a'] = MyStruct(2, 3.3)
    self.assertEqual(td['a'].values, 2)
    self.assertEqual(td['a'].counter, 3.3)
    td['a'].values += 10
    self.assertEqual(td['a'].values, 12)
    self.assertEqual(td['a'].counter, 3.3)
    td['b'] = MyStruct(4, 5.6)