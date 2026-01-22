import array
from collections import namedtuple
import enum
import mmap
import typing as py_typing
import numpy as np
import unittest
from numba.core import types
from numba.core.errors import NumbaValueError, NumbaTypeError
from numba.misc.special import typeof
from numba.core.dispatcher import OmittedArg
from numba._dispatcher import compute_fingerprint
from numba.tests.support import TestCase, skip_unless_cffi, tag
from numba.tests.test_numpy_support import ValueTypingTestBase
from numba.tests.ctypes_usecases import *
from numba.tests.enum_usecases import *
from numba.np import numpy_support
def test_sets(self):
    distinct = DistinctChecker()
    s = compute_fingerprint(set([1]))
    self.assertEqual(compute_fingerprint(set([2, 3])), s)
    distinct.add(s)
    distinct.add(compute_fingerprint([1]))
    distinct.add(compute_fingerprint(set([1j])))
    distinct.add(compute_fingerprint(set([4.5, 6.7])))
    distinct.add(compute_fingerprint(set([(1,)])))
    with self.assertRaises(ValueError):
        compute_fingerprint(set())
    with self.assertRaises(NotImplementedError):
        compute_fingerprint(frozenset([2, 3]))