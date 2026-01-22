from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_tpu_ops
from tensorflow.python.ops.gen_tpu_ops import *
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_function
from tensorflow.python.util.tf_export import tf_export
def enqueue_tpu_embedding_ragged_tensor_batch(sample_splits, embedding_indices, aggregation_weights, table_ids, device_ordinal, max_sequence_lengths=None, num_features=None, combiners=None, mode_override=None, name=None):
    """A placeholder op for enqueueing embedding IDs to the TPU.

  Args:
    sample_splits: A list of rank 1 Tensors specifying the break points for
      splitting embedding_indices and aggregation_weights into rows. It
      corresponds to ids.row_splits in embedding_lookup(), when ids is a
      RaggedTensor. Both int32 and int64 are allowed and will be converted to
      int32 internally.
    embedding_indices: A list of rank 1 Tensors, indices into the embedding
      tables. It corresponds to ids.values in embedding_lookup(), when ids is a
      RaggedTensor. Both int32 and int64 are allowed and will be converted to
      int32 internally.
    aggregation_weights: A list of rank 1 Tensors containing per training
      example aggregation weights. It corresponds to the values field of a
      RaggedTensor with the same row_splits as ids in embedding_lookup(), when
      ids is a RaggedTensor. Both float32 and float64 are allowed and will be
      converted to float32 internally.
    table_ids: A list of integers specifying the identifier of the embedding
      table (offset of TableDescriptor in the TPUEmbeddingConfiguration) to
      lookup the corresponding input. The ith input is looked up using
      table_ids[i]. The size of the table_ids list must be equal to that of
      sample_indices, embedding_indices and aggregation_weights.
    device_ordinal: The TPU device to use. Should be >= 0 and less than the
      number of TPU cores in the task on which the node is placed.
    max_sequence_lengths: A list of integers, the size of which is equal to
      sample_indices. If equal to 0, the corresponding feature is considered to
      be a non-sequence feature, If greater than 0, the corresponding feature is
      a sequence feature with the given maximal length. If None, then we assume
      a list of all zeroes.
    num_features: A list of integers, the size of which must be equal to
      sample_indices. If non-empty, entries in this list must be at least 1. For
      each batch element, we will take num_features rows of the input tensor for
      embedding lookup. E.g., when sample_indices is empty, the embedding
      indices must be of shape (batch_size*num_features).
    combiners: A list of string scalars, one for each embedding table that
      specify how to normalize the embedding activations after weighted
      summation. Supported combiners are 'mean', 'sum', or 'sqrtn'. It is
      invalid to have the sum of the weights be 0 for 'mean' or the sum of the
      squared weights be 0 for 'sqrtn'. If combiners isn't passed, the default
      is to use 'sum' for all tables (optional).
    mode_override: A string input that overrides the mode specified in the
      TPUEmbeddingConfiguration. Supported values are {'unspecified',
      'inference', 'training', 'backward_pass_only'}. When set to 'unspecified',
      the mode set in TPUEmbeddingConfiguration is used, otherwise mode_override
      is used (optional).
    name: A name for the operation (optional).

  Returns:
    An EnqueueTPUEmbeddingRaggedTensorBatch operation.
  """
    if mode_override is None:
        mode_override = 'unspecified'
    return gen_tpu_ops.enqueue_tpu_embedding_ragged_tensor_batch(sample_splits=sample_splits, embedding_indices=embedding_indices, aggregation_weights=aggregation_weights, table_ids=table_ids, device_ordinal=device_ordinal, max_sequence_lengths=max_sequence_lengths, combiners=combiners, mode_override=mode_override, num_features=num_features, name=name)