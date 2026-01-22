import numpy as np
from numba import njit, jit
from numba.core.errors import TypingError
import unittest
from numba.tests.support import TestCase
def test_dict_use_with_none_key(self):

    @njit
    def foo():
        k = {None: 1}
        return k
    with self.assertRaises(TypingError) as raises:
        foo()
    self.assertIn('Dict.key_type cannot be of type none', str(raises.exception))