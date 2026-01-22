import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
@unittest.skipIf(utils.PYVERSION < (3, 9), 'needs Python 3.9+')
def test_unpack_with_predicate_fails(self):

    @njit
    def foo():
        a = (1,)
        b = (3, 2, 4)
        return (*(b if a[0] else (5, 6)),)
    with self.assertRaises(errors.UnsupportedError) as raises:
        foo()
    msg = 'op_LIST_EXTEND at the start of a block'
    self.assertIn(msg, str(raises.exception))