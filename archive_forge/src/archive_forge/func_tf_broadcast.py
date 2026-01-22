import inspect
import numbers
import os
import re
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import flexible_dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.types import core
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export
def tf_broadcast(*args):
    """Broadcast tensors.

  Args:
    *args: a list of tensors whose shapes are broadcastable against each other.

  Returns:
    Tensors broadcasted to the common shape.
  """
    if len(args) <= 1:
        return args
    sh = array_ops.shape(args[0])
    for arg in args[1:]:
        sh = array_ops.broadcast_dynamic_shape(sh, array_ops.shape(arg))
    return [array_ops.broadcast_to(arg, sh) for arg in args]