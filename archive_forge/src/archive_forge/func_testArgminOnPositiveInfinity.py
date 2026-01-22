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
def testArgminOnPositiveInfinity(self, float_type):
    """Ensures we return the right thing for positive infinities."""
    inf = np.array([float('inf')], dtype=np.float32)
    np.testing.assert_equal(np.argmin(inf.astype(float_type)), np.argmin(inf))