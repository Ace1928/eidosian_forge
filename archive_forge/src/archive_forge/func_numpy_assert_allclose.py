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
def numpy_assert_allclose(a, b, float_type, **kwargs):
    a = a.astype(np.float32) if a.dtype == float_type else a
    b = b.astype(np.float32) if b.dtype == float_type else b
    return np.testing.assert_allclose(a, b, **kwargs)