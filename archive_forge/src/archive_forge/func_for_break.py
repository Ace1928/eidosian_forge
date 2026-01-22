import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def for_break(n, x):
    for i in range(n):
        n = 0
        if i == x:
            break
    else:
        n = i
    return (i, n)