import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def var_propagate1(a, b):
    c = (a if a > b else b) + 5
    return c