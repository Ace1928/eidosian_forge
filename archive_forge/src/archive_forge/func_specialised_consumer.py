from numba import njit
from numba.core import errors
from numba.core.extending import overload
import numpy as np
import unittest
@njit
def specialised_consumer(func, *args):
    x, y, z = args
    a = func(x, y, z, mydefault1=1000)
    b = func(x, y, z, mydefault2=1000)
    c = func(x, y, z, mydefault1=1000, mydefault2=1000)
    return a + b + c