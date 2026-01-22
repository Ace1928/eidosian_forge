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
def invalid_isinstance_optional_usecase(x):
    if x > 4:
        z = 10
    else:
        z = None
    if isinstance(z, int):
        return True
    else:
        return False