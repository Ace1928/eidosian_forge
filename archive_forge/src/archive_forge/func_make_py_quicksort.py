import collections
import numpy as np
from numba.core import types
def make_py_quicksort(*args, **kwargs):
    return make_quicksort_impl(lambda f: f, *args, **kwargs)