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
def test_array_values(self):
    """
        Test special.typeof() with ndarray values.
        """

    def check(arr, ndim, layout, mutable, aligned):
        ty = typeof(arr)
        self.assertIsInstance(ty, types.Array)
        self.assertEqual(ty.ndim, ndim)
        self.assertEqual(ty.layout, layout)
        self.assertEqual(ty.mutable, mutable)
        self.assertEqual(ty.aligned, aligned)
    a1 = np.arange(10)
    check(a1, 1, 'C', True, True)
    a2 = np.arange(10).reshape(2, 5)
    check(a2, 2, 'C', True, True)
    check(a2.T, 2, 'F', True, True)
    a3 = np.arange(60)[::2].reshape((2, 5, 3))
    check(a3, 3, 'A', True, True)
    a4 = np.arange(1).reshape(())
    check(a4, 0, 'C', True, True)
    a4.flags.writeable = False
    check(a4, 0, 'C', False, True)
    a5 = a1.astype(a1.dtype.newbyteorder())
    with self.assertRaises(NumbaValueError) as raises:
        typeof(a5)
    self.assertIn('Unsupported array dtype: %s' % (a5.dtype,), str(raises.exception))
    with self.assertRaises(NumbaTypeError) as raises:
        masked_arr = np.ma.MaskedArray([1])
        typeof(masked_arr)
    self.assertIn(f'Unsupported array type: numpy.ma.MaskedArray', str(raises.exception))