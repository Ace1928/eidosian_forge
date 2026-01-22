from numba import int32, int64
from numba import jit
from numba.core import types
from numba.extending import overload
from numba.tests.support import TestCase, tag
import unittest
def generated_inner(x, y=5, z=6):
    assert 0, 'unreachable'