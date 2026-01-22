import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
def input_variations():
    """
            To quote from: https://docs.scipy.org/doc/numpy/reference/generated/numpy.asarray.html    # noqa: E501
            Input data, in any form that can be converted to an array.
            This includes:
            * lists
            * lists of tuples
            * tuples
            * tuples of tuples
            * tuples of lists
            * ndarrays
            """
    yield 1j
    yield 1.2
    yield False
    yield 1
    yield [1, 2, 3]
    yield [(1, 2, 3), (1, 2, 3)]
    yield (1, 2, 3)
    yield ((1, 2, 3), (1, 2, 3))
    yield ([1, 2, 3], [1, 2, 3])
    yield np.array([])
    yield np.arange(4)
    yield np.arange(12).reshape(3, 4)
    yield np.arange(12).reshape(3, 4).T

    def make_list(values):
        a = List()
        for i in values:
            a.append(i)
        return a
    yield make_list((1, 2, 3))
    yield make_list((1.0, 2.0, 3.0))
    yield make_list((1j, 2j, 3j))
    yield make_list((True, False, True))