from contextlib import contextmanager
import sys
import threading
import traceback
from numba.core import types
import numpy as np
from numba.np import numpy_support
from .vector_types import vector_types
def cas(self, array, index, old, val):
    with caslock:
        loaded = array[index]
        if loaded == old:
            array[index] = val
        return loaded