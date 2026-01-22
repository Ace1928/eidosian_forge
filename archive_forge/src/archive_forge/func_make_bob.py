import unittest
import numpy as np
from numba import njit
from numba.core import types
from numba.experimental import structref
from numba.tests.support import skip_unless_scipy
@njit
def make_bob():
    bob = MyStruct('unnamed', vector=np.zeros(3))
    bob.name = 'Bob'
    bob.vector = np.random.random(3)
    return bob