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
def test_staticgetitem_slice(self):

    @njit
    def foo():
        l = ['a', 'b', 1]
        return l[:2]
    with self.assertRaises(errors.TypingError) as raises:
        foo()
    expect = 'Cannot __getitem__ on a literal list'
    self.assertIn(expect, str(raises.exception))