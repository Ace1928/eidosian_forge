import array
import numpy as np
from numba import jit
from numba.tests.support import TestCase, compile_function, MemoryLeakMixin
import unittest
def yield_indices(obj):
    try:
        shape = obj.shape
    except AttributeError:
        shape = (len(obj),)
    for tup in np.ndindex(shape):
        if len(tup) == 1:
            yield tup[0]
        else:
            yield tup