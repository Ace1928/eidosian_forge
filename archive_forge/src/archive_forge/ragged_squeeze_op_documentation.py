from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
Ragged compatible squeeze.

  If `input` is a `tf.Tensor`, then this calls `tf.squeeze`.

  If `input` is a `tf.RaggedTensor`, then this operation takes `O(N)` time,
  where `N` is the number of elements in the squeezed dimensions.

  Args:
    input: A potentially ragged tensor. The input to squeeze.
    axis: An optional list of ints. Defaults to `None`. If the `input` is
      ragged, it only squeezes the dimensions listed. It fails if `input` is
      ragged and axis is []. If `input` is not ragged it calls tf.squeeze. Note
      that it is an error to squeeze a dimension that is not 1. It must be in
      the range of [-rank(input), rank(input)).
   name: A name for the operation (optional).

  Returns:
    A potentially ragged tensor. Contains the same data as input,
    but has one or more dimensions of size 1 removed.
  