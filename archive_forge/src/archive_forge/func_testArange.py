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
def testArange(self, float_type):
    np.testing.assert_equal(np.arange(100, dtype=np.float32).astype(float_type), np.arange(100, dtype=float_type))
    np.testing.assert_equal(np.arange(-8, 8, 1, dtype=np.float32).astype(float_type), np.arange(-8, 8, 1, dtype=float_type))
    np.testing.assert_equal(np.arange(-0.0, -2.0, -0.25, dtype=np.float32).astype(float_type), np.arange(-0.0, -2.0, -0.25, dtype=float_type))
    np.testing.assert_equal(np.arange(-16.0, 16.0, 2.0, dtype=np.float32).astype(float_type), np.arange(-16.0, 16.0, 2.0, dtype=float_type))