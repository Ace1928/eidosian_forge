import itertools
import math
import sys
import unittest
import warnings
import numpy as np
from numba import njit, types
from numba.tests.support import TestCase
from numba.np import numpy_support
def test_gcd(self):
    from itertools import product, repeat, chain
    pyfunc = gcd
    signed_args = product(sorted(types.signed_domain), *repeat((-2, -1, 0, 1, 2, 7, 10), 2))
    unsigned_args = product(sorted(types.unsigned_domain), *repeat((0, 1, 2, 7, 9, 16), 2))
    x_types, x_values, y_values = zip(*chain(signed_args, unsigned_args))
    self.run_binary(pyfunc, x_types, x_values, y_values)