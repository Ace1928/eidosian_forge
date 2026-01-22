import ctypes
import random
from numba.tests.support import TestCase
from numba import _helperlib, jit, typed, types
from numba.core.config import IS_32BITS
from numba.core.datamodel.models import UniTupleModel
from numba.extending import register_model, typeof_impl, unbox, overload
@jit(nopython=True, cache=True)
def set_parametrized_data(x, y):
    dict_vs_cache_vs_parametrized(x)
    dict_vs_cache_vs_parametrized(y)