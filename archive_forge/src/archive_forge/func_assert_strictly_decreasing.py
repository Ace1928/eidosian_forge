import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import uniform as uniform_lib
def assert_strictly_decreasing(array):
    np.testing.assert_array_less(np.diff(array), 0.0)