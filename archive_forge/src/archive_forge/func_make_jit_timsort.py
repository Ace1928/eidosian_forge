import collections
from numba.core import types
def make_jit_timsort(*args):
    from numba import jit
    return make_timsort_impl(lambda f: jit(nopython=True)(f), *args)