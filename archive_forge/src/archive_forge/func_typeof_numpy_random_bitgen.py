from collections import namedtuple
from functools import singledispatch
import ctypes
import enum
import numpy as np
from numpy.random.bit_generator import BitGenerator
from numba.core import types, utils, errors
from numba.np import numpy_support
@typeof_impl.register(BitGenerator)
def typeof_numpy_random_bitgen(val, c):
    return types.NumPyRandomBitGeneratorType(val)