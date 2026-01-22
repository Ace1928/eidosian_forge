from numba import njit
from numba.core import errors
from numba.core.extending import overload
import numpy as np
import unittest
def mul10(x):
    return x * 10