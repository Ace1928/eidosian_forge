import typing
from typing import Protocol
import numpy as np
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import tensor_shape
from tensorflow.python.types import core
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@tf_export('is_tensor')
def is_tf_type(x):
    """Checks whether `x` is a TF-native type that can be passed to many TF ops.

  Use `is_tensor` to differentiate types that can ingested by TensorFlow ops
  without any conversion (e.g., `tf.Tensor`, `tf.SparseTensor`, and
  `tf.RaggedTensor`) from types that need to be converted into tensors before
  they are ingested (e.g., numpy `ndarray` and Python scalars).

  For example, in the following code block:

  ```python
  if not tf.is_tensor(t):
    t = tf.convert_to_tensor(t)
  return t.shape, t.dtype
  ```

  we check to make sure that `t` is a tensor (and convert it if not) before
  accessing its `shape` and `dtype`.  (But note that not all TensorFlow native
  types have shapes or dtypes; `tf.data.Dataset` is an example of a TensorFlow
  native type that has neither shape nor dtype.)

  Args:
    x: A python object to check.

  Returns:
    `True` if `x` is a TensorFlow-native type.
  """
    return isinstance(x, tf_type_classes)