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
def isinstance_dict():
    a = {1: 2, 3: 4}
    b = {'a': 10, 'b': np.zeros(3)}
    if isinstance(a, dict) and isinstance(b, dict):
        return 'dict'
    else:
        return 'not dict'