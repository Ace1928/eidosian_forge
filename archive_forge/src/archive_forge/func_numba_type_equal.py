from numba import vectorize, jit, bool_, double, int_, float_, typeof, int8
import unittest
import numpy as np
def numba_type_equal(a, b):
    self.assertEqual(a.dtype, b.dtype)
    self.assertEqual(a.ndim, b.ndim)