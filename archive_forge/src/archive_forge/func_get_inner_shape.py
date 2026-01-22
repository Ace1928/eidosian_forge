import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def get_inner_shape(item):
    """Returns the inner shape for a python list `item`."""
    if not isinstance(item, (list, tuple)) and np.ndim(item) == 0:
        return ()
    elif len(item) > 0:
        return (len(item),) + get_inner_shape(item[0])
    return (0,)