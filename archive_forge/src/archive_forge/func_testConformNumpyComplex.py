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
@ignore_warning(category=np.ComplexWarning)
def testConformNumpyComplex(self, float_type):
    for dtype in [np.complex64, np.complex128, np.clongdouble]:
        x = np.array([1.5, 2.5 + 2j, 3.5], dtype=dtype)
        y_np = x.astype(np.float32)
        y_tf = x.astype(float_type)
        numpy_assert_allclose(y_np, y_tf, atol=0.02, float_type=float_type)
        z_np = y_np.astype(dtype)
        z_tf = y_tf.astype(dtype)
        numpy_assert_allclose(z_np, z_tf, atol=0.02, float_type=float_type)