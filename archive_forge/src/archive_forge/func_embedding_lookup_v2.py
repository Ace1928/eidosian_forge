from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import data_flow_grad  # pylint: disable=unused-import
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('nn.embedding_lookup', v1=[])
@dispatch.add_dispatch_support
def embedding_lookup_v2(params, ids, max_norm=None, name=None):
    """Looks up embeddings for the given `ids` from a list of tensors.

  This function is used to perform parallel lookups on the list of tensors in
  `params`.  It is a generalization of `tf.gather`, where `params` is
  interpreted as a partitioning of a large embedding tensor.

  If `len(params) > 1`, each element `id` of `ids` is partitioned between the
  elements of `params` according to the "div" partition strategy, which means we
  assign ids to partitions in a contiguous manner. For instance, 13 ids are
  split across 5 partitions as:
  `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`.

  If the id space does not evenly divide the number of partitions, each of the
  first `(max_id + 1) % len(params)` partitions will be assigned one more id.

  The results of the lookup are concatenated into a dense
  tensor. The returned tensor has shape `shape(ids) + shape(params)[1:]`.

  Args:
    params: A single tensor representing the complete embedding tensor, or a
      list of tensors all of same shape except for the first dimension,
      representing sharded embedding tensors following "div" partition strategy.
    ids: A `Tensor` with type `int32` or `int64` containing the ids to be looked
      up in `params`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type as the tensors in `params`.

    For instance, if `params` is a 5x2 matrix:

    ```python
    [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    ```

    or a list of matrices:

    ```python
    params[0]: [[1, 2], [3, 4]]
    params[1]: [[5, 6], [7, 8]]
    params[2]: [[9, 10]]
    ```

    and `ids` is:

    ```python
    [0, 3, 4]
    ```

    The output will be a 3x2 matrix:

    ```python
    [[1, 2], [7, 8], [9, 10]]
    ```

  Raises:
    ValueError: If `params` is empty.
  """
    return embedding_lookup(params, ids, 'div', name, max_norm=max_norm)