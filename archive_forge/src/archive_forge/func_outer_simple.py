from numba import jit, njit
from numba.core import types
from numba.core.extending import overload
def outer_simple(a):
    return inner(a) * 2