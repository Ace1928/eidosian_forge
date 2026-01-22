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
def testCanCast(self, float_type):
    allowed_casts = [(np.bool_, float_type), (np.int8, float_type), (np.uint8, float_type), (float_type, np.float32), (float_type, np.float64), (float_type, np.longdouble), (float_type, np.complex64), (float_type, np.complex128), (float_type, np.clongdouble)]
    all_dtypes = [np.float16, np.float32, np.float64, np.longdouble, np.int8, np.int16, np.int32, np.int64, np.complex64, np.complex128, np.clongdouble, np.uint8, np.uint16, np.uint32, np.uint64, np.intc, np.int_, np.longlong, np.uintc, np.ulonglong]
    for d in all_dtypes:
        with self.subTest(d.__name__):
            self.assertEqual((float_type, d) in allowed_casts, np.can_cast(float_type, d))
            self.assertEqual((d, float_type) in allowed_casts, np.can_cast(d, float_type))