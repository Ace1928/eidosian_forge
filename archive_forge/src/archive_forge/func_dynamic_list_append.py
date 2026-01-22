from tensorflow.python.framework import tensor
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops
def dynamic_list_append(target, element):
    """Converts a list append call inline."""
    if isinstance(target, tensor_array_ops.TensorArray):
        return target.write(target.size(), element)
    if isinstance(target, tensor.Tensor):
        return list_ops.tensor_list_push_back(target, element)
    target.append(element)
    return target