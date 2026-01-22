from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.structured.structured_tensor import _find_shape_dtype
Produce a DynamicRaggedShape for StructuredTensor.