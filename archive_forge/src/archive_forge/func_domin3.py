from numba import njit
from numba.core import types
import unittest
def domin3(a, b, c):
    return min(a, b, c)