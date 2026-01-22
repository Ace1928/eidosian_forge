import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def assignments2(a):
    b = c = d = str(a)
    return b + c + d