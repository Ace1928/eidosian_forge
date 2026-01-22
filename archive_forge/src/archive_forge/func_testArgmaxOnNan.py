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
def testArgmaxOnNan(self, float_type):
    """Ensures we return the right thing for multiple NaNs."""
    one_with_nans = np.array([1.0, float('nan'), float('nan')], dtype=np.float32)
    np.testing.assert_equal(np.argmax(one_with_nans.astype(float_type)), np.argmax(one_with_nans))