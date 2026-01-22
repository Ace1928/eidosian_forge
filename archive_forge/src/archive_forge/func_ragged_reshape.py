from typing import Optional
from typing import Union
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_ragged_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@dispatch.dispatch_for_api(array_ops.reshape)
def ragged_reshape(tensor: ragged_tensor.RaggedOrDense, shape: dynamic_ragged_shape.DenseOrRaggedShape) -> Union[ragged_tensor.RaggedTensor, tensor_lib.Tensor]:
    """Reshapes a tensor or ragged tensor."""
    tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(tensor, name='tensor')
    if isinstance(tensor, ragged_tensor.RaggedTensor):
        tensor = tensor.values
    if isinstance(shape, dynamic_ragged_shape.DynamicRaggedShape):
        flat_values = array_ops.reshape(tensor, shape.inner_shape)
        return ragged_tensor.RaggedTensor._from_nested_row_partitions(flat_values, shape.row_partitions, validate=False)
    else:
        shape = ops.convert_to_tensor(shape, name='shape')
        return array_ops.reshape(tensor, shape)