import functools
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.piecewise_constant_decay', 'train.piecewise_constant'])
def piecewise_constant(x, boundaries, values, name=None):
    """Piecewise constant from boundaries and interval values.

  Example: use a learning rate that's 1.0 for the first 100001 steps, 0.5
    for the next 10000 steps, and 0.1 for any additional steps.

  ```python
  global_step = tf.Variable(0, trainable=False)
  boundaries = [100000, 110000]
  values = [1.0, 0.5, 0.1]
  learning_rate = tf.compat.v1.train.piecewise_constant(global_step, boundaries,
  values)

  # Later, whenever we perform an optimization step, we increment global_step.
  ```

  Args:
    x: A 0-D scalar `Tensor`. Must be one of the following types: `float32`,
      `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`.
    boundaries: A list of `Tensor`s or `int`s or `float`s with strictly
      increasing entries, and with all elements having the same type as `x`.
    values: A list of `Tensor`s or `float`s or `int`s that specifies the values
      for the intervals defined by `boundaries`. It should have one more element
      than `boundaries`, and all elements should have the same type.
    name: A string. Optional name of the operation. Defaults to
      'PiecewiseConstant'.

  Returns:
    A 0-D Tensor. Its value is `values[0]` when `x <= boundaries[0]`,
    `values[1]` when `x > boundaries[0]` and `x <= boundaries[1]`, ...,
    and values[-1] when `x > boundaries[-1]`.

  Raises:
    ValueError: if types of `x` and `boundaries` do not match, or types of all
        `values` do not match or
        the number of elements in the lists does not match.

  @compatibility(eager)
  When eager execution is enabled, this function returns a function which in
  turn returns the decayed learning rate Tensor. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  @end_compatibility
  """
    boundaries = nest.map_structure(tensor_conversion.convert_to_tensor_v2_with_dispatch, nest.flatten(boundaries))
    values = nest.map_structure(tensor_conversion.convert_to_tensor_v2_with_dispatch, nest.flatten(values))
    x_recomp = tensor_conversion.convert_to_tensor_v2_with_dispatch(x)
    for i, b in enumerate(boundaries):
        if b.dtype.base_dtype != x_recomp.dtype.base_dtype:
            if b.dtype.base_dtype == dtypes.int32 and x_recomp.dtype.base_dtype == dtypes.int64:
                b = math_ops.cast(b, x_recomp.dtype.base_dtype)
                boundaries[i] = b
            else:
                raise ValueError('Boundaries (%s) must have the same dtype as x (%s).' % (b.dtype.base_dtype, x_recomp.dtype.base_dtype))
    for v in values[1:]:
        if v.dtype.base_dtype != values[0].dtype.base_dtype:
            raise ValueError('Values must have elements all with the same dtype (%s vs %s).' % (values[0].dtype.base_dtype, v.dtype.base_dtype))
    decayed_lr = learning_rate_schedule.PiecewiseConstantDecay(boundaries, values, name=name)
    if not context.executing_eagerly():
        decayed_lr = decayed_lr(x)
    else:
        decayed_lr = functools.partial(decayed_lr, x)
    return decayed_lr