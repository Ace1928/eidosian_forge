import collections
import functools
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
Slices `linop` along its batch dimensions.

  Args:
    linop: A `LinearOperator` instance.
    params_overrides: A `dict` of parameter overrides.
    slices: A `slice` or `int` or `int` `Tensor` or `tf.newaxis` or `tuple`
      thereof. (e.g. the argument of a `__getitem__` method).

  Returns:
    new_linop: A batch-sliced `LinearOperator`.
  