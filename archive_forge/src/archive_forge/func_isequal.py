import unittest
from numba import jit
from numba.core import types
def isequal(x):
    return x == x