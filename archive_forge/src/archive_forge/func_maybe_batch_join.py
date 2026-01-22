from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.summary import summary
from tensorflow.python.training import queue_runner
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.maybe_batch_join'])
@deprecation.deprecated(None, 'Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.interleave(...).filter(...).batch(batch_size)` (or `padded_batch(...)` if `dynamic_pad=True`).')
def maybe_batch_join(tensors_list, keep_input, batch_size, capacity=32, enqueue_many=False, shapes=None, dynamic_pad=False, allow_smaller_final_batch=False, shared_name=None, name=None):
    """Runs a list of tensors to conditionally fill a queue to create batches.

  See docstring in `batch_join` for more details.

  Args:
    tensors_list: A list of tuples or dictionaries of tensors to enqueue.
    keep_input: A `bool` Tensor.  This tensor controls whether the input is
      added to the queue or not.  If it is a scalar and evaluates `True`, then
      `tensors` are all added to the queue. If it is a vector and `enqueue_many`
      is `True`, then each example is added to the queue only if the
      corresponding value in `keep_input` is `True`. This tensor essentially
      acts as a filtering mechanism.
    batch_size: An integer. The new batch size pulled from the queue.
    capacity: An integer. The maximum number of elements in the queue.
    enqueue_many: Whether each tensor in `tensor_list_list` is a single
      example.
    shapes: (Optional) The shapes for each example.  Defaults to the
      inferred shapes for `tensor_list_list[i]`.
    dynamic_pad: Boolean.  Allow variable dimensions in input shapes.
      The given dimensions are padded upon dequeue so that tensors within a
      batch have the same shapes.
    allow_smaller_final_batch: (Optional) Boolean. If `True`, allow the final
      batch to be smaller if there are insufficient items left in the queue.
    shared_name: (Optional) If set, this queue will be shared under the given
      name across multiple sessions.
    name: (Optional) A name for the operations.

  Returns:
    A list or dictionary of tensors with the same number and types as
    `tensors_list[i]`.

  Raises:
    ValueError: If the `shapes` are not specified, and cannot be
      inferred from the elements of `tensor_list_list`.
  """
    return _batch_join(tensors_list, batch_size, keep_input, capacity=capacity, enqueue_many=enqueue_many, shapes=shapes, dynamic_pad=dynamic_pad, allow_smaller_final_batch=allow_smaller_final_batch, shared_name=shared_name, name=name)