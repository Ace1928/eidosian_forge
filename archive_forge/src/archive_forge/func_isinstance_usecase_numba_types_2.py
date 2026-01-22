import itertools
import functools
import sys
import operator
from collections import namedtuple
import numpy as np
import unittest
import warnings
from numba import jit, typeof, njit, typed
from numba.core import errors, types, config
from numba.tests.support import (TestCase, tag, ignore_internal_warnings,
from numba.core.extending import overload_method, box
def isinstance_usecase_numba_types_2():
    a = b'hello'
    b = range(1, 2)
    c = dict()
    c[2] = 3
    if isinstance(a, bytes) and isinstance(b, range) and isinstance(c, dict):
        return True
    return False