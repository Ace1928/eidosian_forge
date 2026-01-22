from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import tensor_array_ops
def is_tensor_array(t):
    return isinstance(t, tensor_array_ops.TensorArray)