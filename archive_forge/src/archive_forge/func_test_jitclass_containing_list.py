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
def test_jitclass_containing_list(self):
    JCContainer = self.make_jitclass_container()
    expect = Container(n=4)
    got = JCContainer(n=4)
    self.assert_list_element_precise_equal(got.data, expect.data)
    expect.more(3)
    got.more(3)
    self.assert_list_element_precise_equal(got.data, expect.data)