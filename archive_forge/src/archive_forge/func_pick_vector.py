import functools
import hashlib
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util import tf_inspect
def pick_vector(cond, true_vector, false_vector, name='pick_vector'):
    """Picks possibly different length row `Tensor`s based on condition.

  Value `Tensor`s should have exactly one dimension.

  If `cond` is a python Boolean or `tf.constant` then either `true_vector` or
  `false_vector` is immediately returned. I.e., no graph nodes are created and
  no validation happens.

  Args:
    cond: `Tensor`. Must have `dtype=tf.bool` and be scalar.
    true_vector: `Tensor` of one dimension. Returned when cond is `True`.
    false_vector: `Tensor` of one dimension. Returned when cond is `False`.
    name: Python `str`. The name to give this op.
  Example:  ```python pick_vector(tf.less(0, 5), tf.range(10, 12), tf.range(15,
    18))  # [10, 11] pick_vector(tf.less(5, 0), tf.range(10, 12), tf.range(15,
    18))  # [15, 16, 17] ```

  Returns:
    true_or_false_vector: `Tensor`.

  Raises:
    TypeError: if `cond.dtype != tf.bool`
    TypeError: if `cond` is not a constant and
      `true_vector.dtype != false_vector.dtype`
  """
    with ops.name_scope(name, values=(cond, true_vector, false_vector)):
        cond = ops.convert_to_tensor(cond, name='cond')
        if cond.dtype != dtypes.bool:
            raise TypeError('%s.dtype=%s which is not %s' % (cond, cond.dtype, dtypes.bool))
        cond_value_static = tensor_util.constant_value(cond)
        if cond_value_static is not None:
            return true_vector if cond_value_static else false_vector
        true_vector = ops.convert_to_tensor(true_vector, name='true_vector')
        false_vector = ops.convert_to_tensor(false_vector, name='false_vector')
        if true_vector.dtype != false_vector.dtype:
            raise TypeError('%s.dtype=%s does not match %s.dtype=%s' % (true_vector, true_vector.dtype, false_vector, false_vector.dtype))
        n = array_ops.shape(true_vector)[0]
        return array_ops.slice(array_ops.concat([true_vector, false_vector], 0), [array_ops.where_v2(cond, 0, n)], [array_ops.where(cond, n, -1)])