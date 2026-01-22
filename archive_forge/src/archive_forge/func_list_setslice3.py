from collections import namedtuple
import contextlib
import itertools
import math
import sys
import ctypes as ct
import numpy as np
from numba import jit, typeof, njit, literal_unroll, literally
import unittest
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.experimental import jitclass
from numba.core.extending import overload
def list_setslice3(n, start, stop, step):
    l = list(range(n))
    v = l[start:stop:step]
    for i in range(len(v)):
        v[i] += 100
    l[start:stop:step] = v
    return l