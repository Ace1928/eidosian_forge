import contextlib
import threading
from typing import Any, Callable, Optional, Sequence
from tensorflow.dtensor.python import dtensor_device
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.dtensor.is_dtensor', v1=[])
def is_dtensor(tensor) -> bool:
    """Check whether the input tensor is a DTensor.

  In Python, a DTensor has the same type as a `tf.Tensor`. This method will
  let you check and handle the tensor differently if a tf.Tensor is a DTensor.

  Args:
    tensor: an object to be checked.

  Returns:
    bool, True if the given tensor is a DTensor.
  """
    return _dtensor_device().is_dtensor(tensor)