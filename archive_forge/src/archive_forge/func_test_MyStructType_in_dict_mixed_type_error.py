import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
def test_MyStructType_in_dict_mixed_type_error(self):
    self.disable_leak_check()
    td = typed.Dict()
    td['a'] = MyStruct(1, 2.3)
    self.assertEqual(td['a'].values, 1)
    self.assertEqual(td['a'].counter, 2.3)
    with self.assertRaisesRegex(errors.TypingError, 'Cannot cast numba.MyStructType'):
        td['b'] = MyStruct(2.3, 1)