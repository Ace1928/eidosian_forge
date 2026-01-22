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
def binary_operation_test(a, b, op, float_type):
    a = float_type(a)
    b = float_type(b)
    expected = op(np.float32(a), np.float32(b))
    result = op(a, b)
    if math.isnan(expected):
        if not math.isnan(result):
            raise AssertionError('%s expected to be nan.' % repr(result))
    else:
        np.testing.assert_equal(truncate(expected, float_type=float_type), float(result))