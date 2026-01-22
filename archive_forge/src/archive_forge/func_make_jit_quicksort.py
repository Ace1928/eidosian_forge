import collections
import numpy as np
from numba.core import types
def make_jit_quicksort(*args, **kwargs):
    from numba.core.extending import register_jitable
    return make_quicksort_impl(lambda f: register_jitable(f), *args, **kwargs)