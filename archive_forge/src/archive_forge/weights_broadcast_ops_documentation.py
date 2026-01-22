from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sets
from tensorflow.python.util.tf_export import tf_export
Broadcast `weights` to the same shape as `values`.

  This returns a version of `weights` following the same broadcast rules as
  `mul(weights, values)`, but limited to the weights shapes allowed by
  `assert_broadcastable`. When computing a weighted average, use this function
  to broadcast `weights` before summing them; e.g.,
  `reduce_sum(w * v) / reduce_sum(_broadcast_weights(w, v))`.

  Args:
    weights: `Tensor` whose shape is broadcastable to `values` according to the
      rules of `assert_broadcastable`.
    values: `Tensor` of any shape.

  Returns:
    `weights` broadcast to `values` shape according to the rules of
      `assert_broadcastable`.
  