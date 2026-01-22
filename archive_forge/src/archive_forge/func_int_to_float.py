import numpy as np
from numba.core.errors import TypingError
from numba import njit
from numba.core import types
import struct
import unittest
def int_to_float(x):
    return types.float64(x) / 2