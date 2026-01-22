import copy
import enum
import math
from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.feature_column import feature_column_lib as fc_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.feature_column import _is_running_on_cpu
from tensorflow.python.tpu.feature_column import _record_variable_scope_and_name
from tensorflow.python.tpu.feature_column import _SUPPORTED_CATEGORICAL_COLUMNS_V2
from tensorflow.python.tpu.feature_column import _SUPPORTED_SEQUENCE_COLUMNS
from tensorflow.python.tpu.feature_column import _TPUBaseEmbeddingColumn
from tensorflow.python.util.tf_export import tf_export
def sparse_embedding_aggregate_slice(params, values_and_values_mask, combiner='mean', name='sparse_embedding_aggregate_slice'):
    """Uses XLA's dynamic slice operations to perform embedding lookups.

  From third_party/cloud_tpu/models/movielens/tpu_embedding.py

  Args:
    params: Tensor of embedding table. Rank 2 (table_size x embedding dim)
    values_and_values_mask: is a two-tuple that contains: values - Tensor of
      embedding indices. Rank 2 (batch x n_indices) values_mask - Tensor of mask
      / weights. Rank 2 (batch x n_indices)
    combiner: The combiner to use for the embedding lookup. Currently supports
      'sum' and 'mean'.
    name: Optional name scope for created ops

  Returns:
    Rank 2 tensor of aggregated (per batch element) embedding vectors.

  Raises:
    ValueError: Combiner is not supported.
  """
    values, values_mask = values_and_values_mask
    with ops.name_scope(name):
        _, embedding_dimension = params.get_shape().as_list()
        n_batch, n_indices_padded = values.get_shape().as_list()
        if not n_batch:
            n_batch = -1
        emb_lookup = array_ops.reshape(embedding_ops.embedding_lookup(params, array_ops.reshape(values, [n_batch, n_indices_padded])), [n_batch, n_indices_padded, embedding_dimension])
        values_mask_broadcast = array_ops.reshape(values_mask, [n_batch, n_indices_padded, 1])
        aggregate_emb = math_ops.reduce_sum(emb_lookup * values_mask_broadcast, axis=1)
        if combiner == 'sum':
            return aggregate_emb
        elif combiner == 'mean':
            return aggregate_emb / math_ops.maximum(math_ops.reduce_sum(values_mask_broadcast, axis=1), 1.0)
        else:
            raise ValueError('Dense TPU Embedding does not support combiner other than sum and mean.')