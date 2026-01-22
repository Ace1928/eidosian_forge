import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_dispatcher_mistreat(self):

    @jit(nopython=True)
    def foo(x):
        return x
    in1 = (1, 2, 3)
    out1 = foo(in1)
    self.assertEqual(in1, out1)
    in2 = Point(1, 2, 3)
    out2 = foo(in2)
    self.assertEqual(in2, out2)
    self.assertEqual(len(foo.nopython_signatures), 2)
    self.assertEqual(foo.nopython_signatures[0].args[0], typeof(in1))
    self.assertEqual(foo.nopython_signatures[1].args[0], typeof(in2))
    in3 = Point2(1, 2, 3)
    out3 = foo(in3)
    self.assertEqual(in3, out3)
    self.assertEqual(len(foo.nopython_signatures), 3)
    self.assertEqual(foo.nopython_signatures[2].args[0], typeof(in3))