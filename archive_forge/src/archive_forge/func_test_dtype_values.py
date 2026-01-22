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
def test_dtype_values(self):
    self.assertEqual(typeof(np.int64), types.NumberClass(types.int64))
    self.assertEqual(typeof(np.float64), types.NumberClass(types.float64))
    self.assertEqual(typeof(np.int32), types.NumberClass(types.int32))
    self.assertEqual(typeof(np.int8), types.NumberClass(types.int8))