import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import gamma
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@property
def rate(self):
    return self._rate