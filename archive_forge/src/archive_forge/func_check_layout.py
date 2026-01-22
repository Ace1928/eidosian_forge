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
@tf_export('experimental.dtensor.check_layout', v1=[])
def check_layout(tensor: tensor_lib.Tensor, layout: layout_lib.Layout) -> None:
    """Asserts that the layout of the DTensor is `layout`.

  Args:
    tensor: A DTensor whose layout is to be checked.
    layout: The `Layout` to compare against.

  Raises:
    ValueError: If the layout of `tensor` does not match the supplied `layout`.
  """
    if fetch_layout(tensor) != layout:
        raise ValueError('Layout of tensor: ' + str(fetch_layout(tensor)) + ', did not match expected layout: ' + str(layout))