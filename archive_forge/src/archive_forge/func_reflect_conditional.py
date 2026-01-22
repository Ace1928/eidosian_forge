import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def reflect_conditional(sa, sb):
    if len(sb) > 1:
        sa = set((11.0, 22.0, 33.0, 44.0))
    sa.add(42.0)
    sa.update(sb)
    sc = set((55.0, 66.0))
    sa.symmetric_difference_update(sc)
    return (sa, len(sa), len(sb))