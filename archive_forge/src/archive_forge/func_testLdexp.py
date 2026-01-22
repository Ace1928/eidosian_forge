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
@ignore_warning(category=RuntimeWarning, message='invalid value encountered')
def testLdexp(self, float_type):
    rng = np.random.RandomState(seed=42)
    x = rng.randn(3, 7).astype(float_type)
    y = rng.randint(-50, 50, (1, 7)).astype(np.int32)
    self.assertEqual(np.ldexp(x, y).dtype, x.dtype)
    numpy_assert_allclose(np.ldexp(x, y).astype(np.float32), truncate(np.ldexp(x.astype(np.float32), y), float_type=float_type), rtol=0.01, atol=1e-06, float_type=float_type)