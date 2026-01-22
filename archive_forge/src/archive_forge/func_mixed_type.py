import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
@njit
def mixed_type(x, y, m, n):
    return (MyStruct(x, y), MyStruct(m, n))