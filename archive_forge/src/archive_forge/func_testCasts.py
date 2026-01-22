import collections
import contextlib
import copy
import itertools
import math
import pickle
import sys
from typing import Type
import warnings
from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
@ignore_warning(category=RuntimeWarning, message='invalid value encountered in cast')
def testCasts(self, float_type):
    for dtype in [np.float16, np.float32, np.float64, np.longdouble, np.int8, np.int16, np.int32, np.int64, np.complex64, np.complex128, np.clongdouble, np.uint8, np.uint16, np.uint32, np.uint64, np.intc, np.int_, np.longlong, np.uintc, np.ulonglong]:
        x = np.array([[1, 2, 3]], dtype=dtype)
        y = x.astype(float_type)
        z = y.astype(dtype)
        self.assertTrue(np.all(x == y))
        self.assertEqual(float_type, y.dtype)
        self.assertTrue(np.all(x == z))
        self.assertEqual(dtype, z.dtype)