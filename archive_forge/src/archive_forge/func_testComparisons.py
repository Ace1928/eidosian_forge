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
def testComparisons(self, float_type):
    x = np.array([30, 7, -30], dtype=np.float32)
    bx = x.astype(float_type)
    y = np.array([17, 7, 0], dtype=np.float32)
    by = y.astype(float_type)
    np.testing.assert_equal(x == y, bx == by)
    np.testing.assert_equal(x != y, bx != by)
    np.testing.assert_equal(x < y, bx < by)
    np.testing.assert_equal(x > y, bx > by)
    np.testing.assert_equal(x <= y, bx <= by)
    np.testing.assert_equal(x >= y, bx >= by)