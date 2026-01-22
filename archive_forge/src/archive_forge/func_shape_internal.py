import numbers
import numpy as np
from tensorflow.core.config import flags
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import shape_util
from tensorflow.python.ops.gen_array_ops import *
from tensorflow.python.ops.gen_array_ops import reverse_v2 as reverse  # pylint: disable=unused-import
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
def shape_internal(input, name=None, optimize=True, out_type=None):
    """Returns the shape of a tensor.

  If `out_type` is not specified and the shape is fully known, then we look at
  the dimension values to determine whether to return an int32 or int64 tensor.
  If the shape is not fully known, we default to int32.

  Args:
    input: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).
    optimize: if true, encode the shape as a constant when possible.
    out_type: (Optional) The specified output type of the operation (`int32` or
      `int64`). Defaults to tf.int32.

  Returns:
    A `Tensor` of type `out_type`.

  """
    with ops.name_scope(name, 'Shape', [input]) as name:
        if isinstance(input, (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)):
            if not out_type:
                out_type = dtypes.int32
            return gen_math_ops.cast(input.dense_shape, out_type)
        else:
            if not context.executing_eagerly():
                input = ops.convert_to_tensor(input)
                input_shape = input.get_shape()
                if optimize and input_shape.is_fully_defined():
                    if not out_type:
                        return constant_op._tensor_shape_tensor_conversion_function(input_shape)
                    return constant(input_shape.as_list(), out_type, name=name)
            if not out_type:
                out_type = dtypes.int32
            return gen_array_ops.shape(input, name=name, out_type=out_type)