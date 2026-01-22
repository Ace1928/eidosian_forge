from numba import njit
from numba.core import errors
from numba.core.extending import overload
import numpy as np
import unittest
@njit
def consumer2arg(func1, func2):
    return func2(func1)