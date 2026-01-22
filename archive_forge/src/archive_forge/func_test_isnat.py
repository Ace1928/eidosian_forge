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
def test_isnat(self):

    def values():
        yield np.datetime64('2016-01-01')
        yield np.datetime64('NaT')
        yield np.datetime64('NaT', 'ms')
        yield np.datetime64('NaT', 'ns')
        yield np.datetime64('2038-01-19T03:14:07')
        yield np.timedelta64('NaT', 'ms')
        yield np.timedelta64(34, 'ms')
        for unit in ['Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as']:
            yield np.array([123, -321, 'NaT'], dtype='<datetime64[%s]' % unit)
            yield np.array([123, -321, 'NaT'], dtype='<timedelta64[%s]' % unit)
    pyfunc = isnat
    cfunc = jit(nopython=True)(pyfunc)
    for x in values():
        expected = pyfunc(x)
        got = cfunc(x)
        if isinstance(x, np.ndarray):
            self.assertPreciseEqual(expected, got, (x,))
        else:
            self.assertEqual(expected, got, x)