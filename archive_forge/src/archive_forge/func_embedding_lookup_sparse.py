from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch
@dispatch.dispatch_for_api(embedding_ops.embedding_lookup_sparse)
def embedding_lookup_sparse(params, sp_ids: ragged_tensor.Ragged, sp_weights, partition_strategy='mod', name=None, combiner=None, max_norm=None, allow_fast_lookup=False):
    """Looks up embeddings for the given ids and weights from a list of tensors.

  This op assumes that there is at least one id for each row in the dense tensor
  represented by sp_ids (i.e. there are no rows with empty features), and that
  all the indices of sp_ids are in canonical row-major order.

  `sp_ids` and `sp_weights` (if not None) are `RaggedTensor`s with rank of 2.
  Embeddings are always aggregated along the last dimension.

  It also assumes that all id values lie in the range [0, p0), where p0
  is the sum of the size of params along dimension 0.

  Args:
    params: A single tensor representing the complete embedding tensor, or a
      list tensors all of same shape except for the first dimension,
      representing sharded embedding tensors. Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for the given `partition_strategy`.
    sp_ids: `RaggedTensor` with rank 2. The rank is not verified for performance
      reasons.
    sparse_weights: `RaggedTensor` of same type and shape as `sparse_ids`,
      containing float / double weights corresponding to `sparse_ids`, or `None`
      if all weights are assumed to be 1.0.
    partition_strategy: A string specifying the partitioning strategy, relevant
      if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
      is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: Optional name for the op.
    combiner: A string specifying the reduction op. Currently "mean", "sqrtn"
      and "sum" are supported. "sum" computes the weighted sum of the embedding
      results for each row. "mean" is the weighted sum divided by the total
      weight. "sqrtn" is the weighted sum divided by the square root of the sum
      of the squares of the weights. Defaults to `mean`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value, before combining.
    allow_fast_lookup: An optional boolean specifying whether to allow
      simplified embedding lookups when `params` is a single tensor and
      `max_norm` is `None`. Setting this flag to `True` during training can
      cause the use of dense gradients with increased memory footprint.

  Returns:
    A dense tensor representing the combined embeddings for the
    sparse ids. For each row in the dense tensor represented by `sp_ids`, the op
    looks up the embeddings for all ids in that row, multiplies them by the
    corresponding weight, and combines these embeddings as specified.

    In other words, if

      `shape(combined params) = [p0, p1, ..., pm]`

    and

      `shape(sp_ids) = shape(sp_weights) = [d0, d1]`

    then

      `shape(output) = [d0, p1, ..., pm]`.

    For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are

      ```python
      [0, 0]: id 1, weight 2.0
      [0, 1]: id 3, weight 0.5
      [1, 0]: id 0, weight 1.0
      [2, 3]: id 1, weight 3.0
      ```

    with `combiner`="mean", then the output will be a 3x20 matrix where

      ```python
      output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
      output[1, :] = (params[0, :] * 1.0) / 1.0
      output[2, :] = (params[1, :] * 3.0) / 3.0
      ```

  Raises:
    TypeError: If `sp_weights` is neither `None` nor of the same type as
      `sp_ids`.
    ValueError: If `combiner` is not one of {"mean", "sqrtn", "sum"}.
  """
    rt_ids = sp_ids
    rt_weights = sp_weights
    if combiner is None:
        combiner = 'mean'
    if combiner not in ('mean', 'sqrtn', 'sum'):
        raise ValueError(f"combiner must be one of 'mean', 'sqrtn' or 'sum', got {combiner}")
    if isinstance(params, variables.PartitionedVariable):
        params = list(params)
    if not isinstance(params, list):
        params = [params]
    ignore_weights = rt_weights is None
    if not ignore_weights:
        if not isinstance(rt_weights, ragged_tensor.RaggedTensor):
            raise TypeError(f'sp_ids must be of the same type as sp_weights, received {{type(sp_ids).__name__!r}} for sp_ids and {{type(sp_weights).__name__!r}} for sp_weights.')
        rt_ids.values.get_shape().assert_is_compatible_with(rt_weights.values.get_shape())
        rt_ids.get_shape().assert_is_compatible_with(rt_weights.get_shape())
    with ops.name_scope(name, 'embedding_lookup_sparse', params + [rt_ids]) as name:
        segment_ids = rt_ids.value_rowids()
        ids = rt_ids.flat_values
        return embedding_ops.embedding_lookup_sparse_impl(params, segment_ids, sp_weights, ids, combiner, ignore_weights, max_norm, allow_fast_lookup, partition_strategy, name)