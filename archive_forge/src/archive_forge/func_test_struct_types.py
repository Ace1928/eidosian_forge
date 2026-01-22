import sys
from itertools import product
import numpy as np
import unittest
from numba.core import types
from numba.core.errors import NumbaNotImplementedError
from numba.tests.support import TestCase
from numba.tests.enum_usecases import Shake, RequestError
from numba.np import numpy_support
def test_struct_types(self):

    def check(dtype, fields, size, aligned):
        tp = numpy_support.from_dtype(dtype)
        self.assertIsInstance(tp, types.Record)
        self.assertEqual(tp.dtype, dtype)
        self.assertEqual(tp.fields, fields)
        self.assertEqual(tp.size, size)
        self.assertEqual(tp.aligned, aligned)
    dtype = np.dtype([('a', np.int16), ('b', np.int32)])
    check(dtype, fields={'a': (types.int16, 0, None, None), 'b': (types.int32, 2, None, None)}, size=6, aligned=False)
    dtype = np.dtype([('a', np.int16), ('b', np.int32)], align=True)
    check(dtype, fields={'a': (types.int16, 0, None, None), 'b': (types.int32, 4, None, None)}, size=8, aligned=True)
    dtype = np.dtype([('m', np.int32), ('n', 'S5')])
    check(dtype, fields={'m': (types.int32, 0, None, None), 'n': (types.CharSeq(5), 4, None, None)}, size=9, aligned=False)