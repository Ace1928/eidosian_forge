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
@tf_export(v1=['tpu.experimental.embedding_column'])
def embedding_column_v2(categorical_column, dimension, combiner='mean', initializer=None, max_sequence_length=0, learning_rate_fn=None, embedding_lookup_device=None, tensor_core_shape=None, use_safe_embedding_lookup=True):
    """TPU version of `tf.compat.v1.feature_column.embedding_column`.

  Note that the interface for `tf.tpu.experimental.embedding_column` is
  different from that of `tf.compat.v1.feature_column.embedding_column`: The
  following arguments are NOT supported: `ckpt_to_load_from`,
  `tensor_name_in_ckpt`, `max_norm` and `trainable`.

  Use this function in place of `tf.compat.v1.feature_column.embedding_column`
  when you want to use the TPU to accelerate your embedding lookups via TPU
  embeddings.

  ```
  column = tf.feature_column.categorical_column_with_identity(...)
  tpu_column = tf.tpu.experimental.embedding_column(column, 10)
  ...
  def model_fn(features):
    dense_feature = tf.keras.layers.DenseFeature(tpu_column)
    embedded_feature = dense_feature(features)
    ...

  estimator = tf.estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      ...
      embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
        column=[tpu_column],
        ...))
  ```

  Args:
    categorical_column: A categorical column returned from
        `categorical_column_with_identity`, `weighted_categorical_column`,
        `categorical_column_with_vocabulary_file`,
        `categorical_column_with_vocabulary_list`,
        `sequence_categorical_column_with_identity`,
        `sequence_categorical_column_with_vocabulary_file`,
        `sequence_categorical_column_with_vocabulary_list`
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row for a non-sequence column. For more information, see
      `tf.feature_column.embedding_column`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.compat.v1.truncated_normal_initializer` with mean `0.0` and
      standard deviation `1/sqrt(dimension)`.
    max_sequence_length: An non-negative integer specifying the max sequence
      length. Any sequence shorter then this will be padded with 0 embeddings
      and any sequence longer will be truncated. This must be positive for
      sequence features and 0 for non-sequence features.
    learning_rate_fn: A function that takes global step and returns learning
      rate for the embedding table. If you intend to use the same learning rate
      for multiple embedding tables, please ensure that you pass the exact same
      python function to all calls of embedding_column, otherwise performence
      may suffer.
    embedding_lookup_device: The device on which to run the embedding lookup.
      Valid options are "cpu", "tpu_tensor_core", and "tpu_embedding_core".
      If specifying "tpu_tensor_core", a tensor_core_shape must be supplied.
      If not specified, the default behavior is embedding lookup on
      "tpu_embedding_core" for training and "cpu" for inference.
      Valid options for training : ["tpu_embedding_core", "tpu_tensor_core"]
      Valid options for serving :  ["cpu", "tpu_tensor_core"]
      For training, tpu_embedding_core is good for large embedding vocab (>1M),
      otherwise, tpu_tensor_core is often sufficient.
      For serving, doing embedding lookup on tpu_tensor_core during serving is
      a way to reduce host cpu usage in cases where that is a bottleneck.
    tensor_core_shape: If supplied, a list of integers which specifies
      the intended dense shape to run embedding lookup for this feature on
      TensorCore. The batch dimension can be left None or -1 to indicate
      a dynamic shape. Only rank 2 shapes currently supported.
    use_safe_embedding_lookup: If true, uses safe_embedding_lookup_sparse
      instead of embedding_lookup_sparse. safe_embedding_lookup_sparse ensures
      there are no empty rows and all weights and ids are positive at the
      expense of extra compute cost. This only applies to rank 2 (NxM) shaped
      input tensors. Defaults to true, consider turning off if the above checks
      are not needed. Note that having empty rows will not trigger any error
      though the output result might be 0 or omitted.

  Returns:
    A  `_TPUEmbeddingColumnV2`.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if `initializer` is specified but not callable.
  """
    if not isinstance(categorical_column, _SUPPORTED_CATEGORICAL_COLUMNS_V2):
        raise TypeError('categorical_column for tpu embedding_column must be type {}, got {}.'.format(' or '.join([cc.__name__ for cc in _SUPPORTED_CATEGORICAL_COLUMNS_V2]), type(categorical_column)))
    if dimension is None or dimension < 1:
        raise ValueError('Invalid dimension {}.'.format(dimension))
    if tensor_core_shape and len(tensor_core_shape) != 2:
        raise ValueError('tensor_core_shape must be size 2. Got {}.'.format(tensor_core_shape))
    if initializer is not None and (not callable(initializer)):
        raise ValueError('initializer must be callable if specified. Embedding of column_name: {}'.format(categorical_column.name))
    if initializer is None:
        initializer = init_ops.truncated_normal_initializer(mean=0.0, stddev=1 / math.sqrt(dimension))
    if embedding_lookup_device and embedding_lookup_device not in _ALLOWED_DEVICES:
        raise ValueError(f'If set, embedding_lookup_device must be in {_ALLOWED_DEVICES}')
    if embedding_lookup_device == 'cpu':
        embedding_lookup_device = EmbeddingDevice.CPU
    elif embedding_lookup_device == 'tpu_tensor_core':
        embedding_lookup_device = EmbeddingDevice.TPU_TENSOR_CORE
    elif embedding_lookup_device == 'tpu_embedding_core':
        embedding_lookup_device = EmbeddingDevice.TPU_EMBEDDING_CORE
    if embedding_lookup_device == EmbeddingDevice.TPU_TENSOR_CORE:
        if not tensor_core_shape:
            raise ValueError('Using embedding_lookup_device=tpu_tensor_core requires tensor_core_shape to be set.')
        if isinstance(categorical_column, _SUPPORTED_SEQUENCE_COLUMNS):
            raise ValueError('embedding_lookup_device=tpu_tensor_core currently does not support sequence columns.')
    if not embedding_lookup_device:
        return _TPUEmbeddingColumnV2(categorical_column=categorical_column, dimension=dimension, combiner=combiner, initializer=initializer, max_sequence_length=max_sequence_length, learning_rate_fn=learning_rate_fn, use_safe_embedding_lookup=use_safe_embedding_lookup)
    else:
        return _TPUDeviceSpecificEmbeddingColumnV2(categorical_column=categorical_column, dimension=dimension, combiner=combiner, initializer=initializer, max_sequence_length=max_sequence_length, learning_rate_fn=learning_rate_fn, embedding_lookup_device=embedding_lookup_device, tensor_core_shape=tensor_core_shape, use_safe_embedding_lookup=use_safe_embedding_lookup)