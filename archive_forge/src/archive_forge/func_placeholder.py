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
@tf_export(v1=['ragged.placeholder'])
@dispatch.add_dispatch_support
def placeholder(dtype, ragged_rank, value_shape=None, name=None):
    """Creates a placeholder for a `tf.RaggedTensor` that will always be fed.

  **Important**: This ragged tensor will produce an error if evaluated.
  Its value must be fed using the `feed_dict` optional argument to
  `Session.run()`, `Tensor.eval()`, or `Operation.run()`.


  Args:
    dtype: The data type for the `RaggedTensor`.
    ragged_rank: The ragged rank for the `RaggedTensor`
    value_shape: The shape for individual flat values in the `RaggedTensor`.
    name: A name for the operation (optional).

  Returns:
    A `RaggedTensor` that may be used as a handle for feeding a value, but
    not evaluated directly.

  Raises:
    RuntimeError: if eager execution is enabled

  @compatibility(TF2)
  This API is not compatible with eager execution and `tf.function`. To migrate
  to TF2, rewrite the code to be compatible with eager execution. Check the
  [migration
  guide](https://www.tensorflow.org/guide/migrate#1_replace_v1sessionrun_calls)
  on replacing `Session.run` calls. In TF2, you can just pass tensors directly
  into ops and layers. If you want to explicitly set up your inputs, also see
  [Keras functional API](https://www.tensorflow.org/guide/keras/functional) on
  how to use `tf.keras.Input` to replace `tf.compat.v1.ragged.placeholder`.
  `tf.function` arguments also do the job of `tf.compat.v1.ragged.placeholder`.
  For more details please read [Better
  performance with tf.function](https://www.tensorflow.org/guide/function).
  @end_compatibility
  """
    if ragged_rank == 0:
        return array_ops.placeholder(dtype, value_shape, name)
    with ops.name_scope(name, 'RaggedPlaceholder', []):
        flat_shape = tensor_shape.TensorShape([None]).concatenate(value_shape)
        result = array_ops.placeholder(dtype, flat_shape, 'flat_values')
        for i in reversed(range(ragged_rank)):
            row_splits = array_ops.placeholder(dtypes.int64, [None], 'row_splits_%d' % i)
            result = ragged_tensor.RaggedTensor.from_row_splits(result, row_splits, validate=False)
        return result