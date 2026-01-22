import numpy as np
from numba import njit, jit
from numba.core.errors import TypingError
import unittest
from numba.tests.support import TestCase
def test_dict_use_with_optional_value(self):

    @njit
    def foo(choice):
        optional = 2.5 if choice else None
        k = {1: optional}
        return k
    with self.assertRaises(TypingError) as raises:
        foo(True)
    self.assertIn('Dict.value_type cannot be of type OptionalType(float64)', str(raises.exception))