from numba import njit
from numba.core import errors
from numba.core.extending import overload
import numpy as np
import unittest
def impl_factory(consumer_func):

    def impl():
        t = 12

        def inner(a, b, c, mydefault1=123, mydefault2=456):
            z = 4
            return mydefault1 + mydefault2 + z + t + a + b + c
        return (inner(1, 2, 5, 91, 53), consumer_func(inner, 1, 2, 11), consumer_func(inner, 1, 2, 3), inner(1, 2, 4))
    return impl