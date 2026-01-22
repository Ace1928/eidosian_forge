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
@dispatch.dispatch_for_api(embedding_ops.safe_embedding_lookup_sparse)
def safe_embedding_lookup_sparse(embedding_weights, sparse_ids: ragged_tensor.Ragged, sparse_weights=None, combiner='mean', default_id=None, name=None, partition_strategy='div', max_norm=None, allow_fast_lookup=False):
    """Lookup embedding results, accounting for invalid IDs and empty features.

  The partitioned embedding in `embedding_weights` must all be the same shape
  except for the first dimension. The first dimension is allowed to vary as the
  vocabulary size is not necessarily a multiple of `P`.  `embedding_weights`
  may be a `PartitionedVariable` as returned by using
  `tf.compat.v1.get_variable()` with a
  partitioner.

  Invalid IDs (< 0) are pruned from input IDs and weights, as well as any IDs
  with non-positive weight. For an entry with no features, the embedding vector
  for `default_id` is returned, or the 0-vector if `default_id` is not supplied.

  The ids and weights may be multi-dimensional `SparseTensor`s or
  `RaggedTensor`s with rank of 2. For `SpareTensor`s with left-aligned non-zero
  entries which can be described as `RaggedTensor`s, use of `RaggedTensor`s can
  yield higher performance. Embeddings are always aggregated along the last
  dimension.

  Args:
    embedding_weights: A single tensor representing the complete embedding
      tensor, or a list tensors all of same shape except for the first
      dimension, representing sharded embedding tensors. Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for the given `partition_strategy`.
    sp_ids: `RaggedTensor` with rank 2. The rank is not verified for performance
      reasons.
    sparse_weights: `RaggedTensor` of same type and shape as `sparse_ids`,
      containing float weights corresponding to `sparse_ids`, or `None` if all
      weights are assumed to be 1.0.
    combiner: A string specifying how to combine embedding results for each
      entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean" the
      default.
    default_id: The id to use for an entry with no features.
    name: A name for this operation (optional).
    partition_strategy: A string specifying the partitioning strategy. Currently
      `"div"` and `"mod"` are supported. Default is `"div"`.
    max_norm: If not `None`, all embeddings are l2-normalized to max_norm before
      combining.
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

      `shape(combined embedding_weights) = [p0, p1, ..., pm]`

    and

      `shape(sparse_ids) = shape(sparse_weights) = [d0, d1, ..., dn]`

    then

      `shape(output) = [d0, d1, ... dn-1, p1, ..., pm]`.

    For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are

      ```python
      [0, 0]: id 1, weight 2.0
      [0, 1]: id 3, weight 0.5
      [1, 0]: id -1, weight 1.0
      [2, 3]: id 1, weight 3.0
      ```

    `default_id` is 0.

    with `combiner`="mean", then the output will be a 3x20 matrix where

      ```python
      output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
      output[1, :] = (params[0, :] * 1.0) / 1.0
      output[2, :] = (params[1, :] * 3.0) / 3.0
      ```

  Raises:
    ValueError: if `embedding_weights` is empty.
  """
    ragged_ids = sparse_ids
    ragged_weights = sparse_weights
    if embedding_weights is None:
        raise ValueError(f'Missing embedding_weights {embedding_weights}.')
    if isinstance(embedding_weights, variables.PartitionedVariable):
        embedding_weights = list(embedding_weights)
    if not isinstance(embedding_weights, list):
        embedding_weights = [embedding_weights]
    if len(embedding_weights) < 1:
        raise ValueError(f'Missing embedding_weights {embedding_weights}.')
    dtype = ragged_weights.dtype if ragged_weights is not None else None
    embedding_weights = [w if isinstance(w, resource_variable_ops.ResourceVariable) and dtype in (None, w.dtype) else ops.convert_to_tensor(w, dtype=dtype) for w in embedding_weights]
    with ops.name_scope(name, 'embedding_lookup', embedding_weights + [ragged_ids, ragged_weights]) as scope:
        ragged_ids, ragged_weights = _prune_invalid_ids_ragged(ragged_ids, ragged_weights)
        if combiner != 'sum':
            ragged_ids, ragged_weights = _prune_invalid_weights_ragged(ragged_ids, ragged_weights)
        ragged_ids, is_row_empty = ragged_array_ops.fill_empty_rows(ragged_ids, default_id or 0)
        if ragged_weights is not None:
            ragged_weights, _ = ragged_array_ops.fill_empty_rows(ragged_weights, 1.0)
        result = embedding_lookup_sparse(embedding_weights, ragged_ids, ragged_weights, combiner=combiner, partition_strategy=partition_strategy, name=None if default_id is None else scope, max_norm=max_norm, allow_fast_lookup=allow_fast_lookup)
        if default_id is None:
            is_row_empty = array_ops.tile(array_ops.reshape(is_row_empty, [-1, 1]), array_ops_stack.stack([1, array_ops.shape(result)[1]]))
            result = array_ops.where(is_row_empty, array_ops.zeros_like(result), result, name=scope)
        return result