from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def full_test_arrays(dtype):
    array_list = base_test_arrays(dtype)
    if dtype == np.float32:
        array_list += [a / 10 for a in array_list]
    if dtype == np.complex64:
        acc = []
        for a in array_list:
            tmp = a / 10 + 1j * a / 11
            tmp[::2] = np.conj(tmp[::2])
            acc.append(tmp)
        array_list.extend(acc)
    for a in array_list:
        assert a.dtype == np.dtype(dtype)
    return array_list