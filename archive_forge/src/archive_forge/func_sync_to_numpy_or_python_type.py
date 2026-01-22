import collections
import copy
import numpy as np
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.distribute.coordinator import cluster_coordinator as coordinator_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.util import nest
def sync_to_numpy_or_python_type(tensors):
    """Syncs and converts a structure of `Tensor`s to `NumPy` arrays or Python scalar types.

  For each tensor, it calls `tensor.numpy()`. If the result is a scalar value,
  it converts it to a Python type, such as a float or int, by calling
  `result.item()`.

  Numpy scalars are converted, as Python types are often more convenient to deal
  with. This is especially useful for bfloat16 Numpy scalars, which don't
  support as many operations as other Numpy values.

  Async strategies (such as `TPUStrategy` and `ParameterServerStrategy`) are
  forced to
  sync during this process.

  Args:
    tensors: A structure of tensors.

  Returns:
    `tensors`, but scalar tensors are converted to Python types and non-scalar
    tensors are converted to Numpy arrays.
  """
    if isinstance(tensors, coordinator_lib.RemoteValue):
        return tensors.fetch()

    def _to_single_numpy_or_python_type(t):
        if isinstance(t, tensor_lib.Tensor):
            x = t.numpy()
            return x.item() if np.ndim(x) == 0 else x
        return t
    return nest.map_structure(_to_single_numpy_or_python_type, tensors)