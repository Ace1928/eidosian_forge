import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def chained_compare(a):
    return 1 < a < 3