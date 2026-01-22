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
def list_extend_heterogeneous(n):
    l = []
    l.extend(range(n))
    l.extend(l[:-1])
    l.extend((5, 42))
    l.extend([123.0])
    return l