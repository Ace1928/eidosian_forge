import copy
import enum
import functools
import sys
import threading
import traceback
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def single_slice_dim(self, shape):
    """Returns the slice dim when the variable is partitioned only in one dim.

    Args:
      shape: Tuple or list of `int` indicating the shape of one specific
        variable partition.

    Returns:
      `int` representing the dimension that the variable is partitioned in, or
      `None` if the variable doesn't seem to be partitioned at all.

    Raises:
      TypeError: If `shape` is not a sequence.
      ValueError: If `shape` is not the same length as `self.full_shape`. If
        the variable is partitioned in more than one dimension.
    """
    if not isinstance(shape, (tuple, list)):
        raise TypeError('`shape` must be a sequence (like tuple or list) instead of ' + type(shape).__name__)
    if len(shape) != len(self.full_shape):
        raise ValueError('Expected equal length, but received shape={} of length {} while self.full_shape={} is of length {}.'.format(shape, len(shape), self.full_shape, len(self.full_shape)))
    for i in range(len(shape)):
        if self.var_offset[i] + shape[i] > self.full_shape[i]:
            raise ValueError('With self.var_offset={}, a partition of shape={} would exceed self.full_shape={} in dimension {}.'.format(self.var_offset, shape, self.full_shape, i))
    slice_dim = None
    for i in range(len(shape)):
        if shape[i] == self.full_shape[i]:
            continue
        if slice_dim is not None:
            raise ValueError('Cannot use single_slice_dim() with shape={} and self.full_shape={} since slice dim could be either dimension {} or {}.'.format(shape, self.full_shape, i, slice_dim))
        slice_dim = i
    return slice_dim