import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
def global_int_tuple():
    return tup_int[0] + tup_int[1]