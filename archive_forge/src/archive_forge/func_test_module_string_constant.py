import math
import sys
import numpy as np
from numba import njit
import numba.tests.usecases as uc
import unittest
def test_module_string_constant(self):

    @njit
    def f():
        return uc._GLOBAL_STR
    self.assertEqual(f(), f.py_func())