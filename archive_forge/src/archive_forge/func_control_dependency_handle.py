import contextlib
from tensorflow.python.framework import ops
from tensorflow.python.ops import tensor_array_ops
def control_dependency_handle(t):
    if isinstance(t, tensor_array_ops.TensorArray):
        return t.flow
    return t