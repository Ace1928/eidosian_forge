from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_tpu_ops
from tensorflow.python.ops.gen_tpu_ops import *
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_function
from tensorflow.python.util.tf_export import tf_export
def infeed_dequeue(dtype, shape, name=None):
    """A placeholder op for a value that will be fed into the computation.

  Args:
    dtype: A `tf.DType`. The type of elements in the tensor.
    shape: A `tf.TensorShape` or list of `ints`. The shape of the tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
    A tensor that will be provided using the infeed mechanism.

  Raises:
    TypeError: If 'dtype` is not a supported infeed type.
  """
    if dtype not in _SUPPORTED_INFEED_DTYPES:
        raise TypeError("Operation '{}' has type {} which is not a supported TPU infeed type. Supported types are: {}".format(name, dtype, list(_SUPPORTED_INFEED_DTYPES)))
    return gen_tpu_ops.infeed_dequeue(dtype, shape, name=name)