from contextlib import contextmanager
import sys
import threading
import traceback
from numba.core import types
import numpy as np
from numba.np import numpy_support
from .vector_types import vector_types
def compare_and_swap(self, array, old, val):
    with compare_and_swaplock:
        index = (0,) * array.ndim
        loaded = array[index]
        if loaded == old:
            array[index] = val
        return loaded