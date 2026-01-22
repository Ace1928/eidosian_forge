import math
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import util as losses_util
from tensorflow.python.platform import device_context
from tensorflow.python.util import dispatch
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
@tf_export('math.zero_fraction', 'nn.zero_fraction')
@dispatch.add_dispatch_support
def zero_fraction(value, name=None):
    """Returns the fraction of zeros in `value`.

  If `value` is empty, the result is `nan`.

  This is useful in summaries to measure and report sparsity.  For example,

  ```python
      z = tf.nn.relu(...)
      summ = tf.compat.v1.summary.scalar('sparsity', tf.nn.zero_fraction(z))
  ```

  Args:
    value: A tensor of numeric type.
    name: A name for the operation (optional).

  Returns:
    The fraction of zeros in `value`, with type `float32`.
  """
    with ops.name_scope(name, 'zero_fraction', [value]):
        value = ops.convert_to_tensor(value, name='value')
        size = array_ops.size(value, out_type=dtypes.int64)
        num_nonzero = tf_cond.cond(size <= dtypes.int32.max, true_fn=lambda: math_ops.cast(_count_nonzero(value, dtype=dtypes.int32), dtype=dtypes.int64), false_fn=lambda: _count_nonzero(value, dtype=dtypes.int64))
        with ops.name_scope('counts_to_fraction'):
            num_zero = size - num_nonzero
            num_zero_float32 = math_ops.cast(num_zero, dtype=dtypes.float32)
            size_float32 = math_ops.cast(size, dtype=dtypes.float32)
            zero_fraction_float32 = num_zero_float32 / size_float32
        return array_ops.identity(zero_fraction_float32, 'fraction')