import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
def dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch(sample_indices_or_row_splits: List[_atypes.TensorFuzzingAnnotation[TV_DynamicEnqueueTPUEmbeddingArbitraryTensorBatch_T1]], embedding_indices: List[_atypes.TensorFuzzingAnnotation[TV_DynamicEnqueueTPUEmbeddingArbitraryTensorBatch_T2]], aggregation_weights: List[_atypes.TensorFuzzingAnnotation[TV_DynamicEnqueueTPUEmbeddingArbitraryTensorBatch_T3]], mode_override: _atypes.TensorFuzzingAnnotation[_atypes.String], device_ordinal: _atypes.TensorFuzzingAnnotation[_atypes.Int32], combiners=[], name=None):
    """Eases the porting of code that uses tf.nn.embedding_lookup_sparse().

  embedding_indices[i] and aggregation_weights[i] correspond
  to the ith feature.

  The tensors at corresponding positions in the three input lists (sample_indices,
  embedding_indices and aggregation_weights) must have the same shape, i.e. rank 1
  with dim_size() equal to the total number of lookups into the table described by
  the corresponding feature.

  Args:
    sample_indices_or_row_splits: A list of at least 1 `Tensor` objects with the same type in: `int32`, `int64`.
      A list of rank 2 Tensors specifying the training example to which the
      corresponding embedding_indices and aggregation_weights values belong.
      If the size of its first dimension is 0, we assume each embedding_indices
      belongs to a different sample. Both int32 and int64 are allowed and will
      be converted to int32 internally.

      Or a list of rank 1 Tensors specifying the row splits for splitting
      embedding_indices and aggregation_weights into rows. It corresponds to
      ids.row_splits in embedding_lookup(), when ids is a RaggedTensor. When
      enqueuing N-D ragged tensor, only the last dimension is allowed to be ragged.
      the row splits is 1-D dense tensor. When empty, we assume a dense tensor is
      passed to the op Both int32 and int64 are allowed and will be converted to
      int32 internally.
    embedding_indices: A list with the same length as `sample_indices_or_row_splits` of `Tensor` objects with the same type in: `int32`, `int64`.
      A list of rank 1 Tensors, indices into the embedding
      tables. Both int32 and int64 are allowed and will be converted to
      int32 internally.
    aggregation_weights: A list with the same length as `sample_indices_or_row_splits` of `Tensor` objects with the same type in: `float32`, `float64`.
      A list of rank 1 Tensors containing per training
      example aggregation weights. Both float32 and float64 are allowed and will
      be converted to float32 internally.
    mode_override: A `Tensor` of type `string`.
      A string input that overrides the mode specified in the
      TPUEmbeddingConfiguration. Supported values are {'unspecified', 'inference',
      'training', 'backward_pass_only'}. When set to 'unspecified', the mode set
      in TPUEmbeddingConfiguration is used, otherwise mode_override is used.
    device_ordinal: A `Tensor` of type `int32`.
      The TPU device to use. Should be >= 0 and less than the number
      of TPU cores in the task on which the node is placed.
    combiners: An optional list of `strings`. Defaults to `[]`.
      A list of string scalars, one for each embedding table that specify
      how to normalize the embedding activations after weighted summation.
      Supported combiners are 'mean', 'sum', or 'sqrtn'. It is invalid to have
      the sum of the weights be 0 for 'mean' or the sum of the squared weights be
      0 for 'sqrtn'. If combiners isn't passed, the default is to use 'sum' for
      all tables.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DynamicEnqueueTPUEmbeddingArbitraryTensorBatch', name, sample_indices_or_row_splits, embedding_indices, aggregation_weights, mode_override, device_ordinal, 'combiners', combiners)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch_eager_fallback(sample_indices_or_row_splits, embedding_indices, aggregation_weights, mode_override, device_ordinal, combiners=combiners, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(sample_indices_or_row_splits, (list, tuple)):
        raise TypeError("Expected list for 'sample_indices_or_row_splits' argument to 'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % sample_indices_or_row_splits)
    _attr_N = len(sample_indices_or_row_splits)
    if not isinstance(embedding_indices, (list, tuple)):
        raise TypeError("Expected list for 'embedding_indices' argument to 'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % embedding_indices)
    if len(embedding_indices) != _attr_N:
        raise ValueError("List argument 'embedding_indices' to 'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op with length %d must match length %d of argument 'sample_indices_or_row_splits'." % (len(embedding_indices), _attr_N))
    if not isinstance(aggregation_weights, (list, tuple)):
        raise TypeError("Expected list for 'aggregation_weights' argument to 'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % aggregation_weights)
    if len(aggregation_weights) != _attr_N:
        raise ValueError("List argument 'aggregation_weights' to 'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op with length %d must match length %d of argument 'sample_indices_or_row_splits'." % (len(aggregation_weights), _attr_N))
    if combiners is None:
        combiners = []
    if not isinstance(combiners, (list, tuple)):
        raise TypeError("Expected list for 'combiners' argument to 'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % combiners)
    combiners = [_execute.make_str(_s, 'combiners') for _s in combiners]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('DynamicEnqueueTPUEmbeddingArbitraryTensorBatch', sample_indices_or_row_splits=sample_indices_or_row_splits, embedding_indices=embedding_indices, aggregation_weights=aggregation_weights, mode_override=mode_override, device_ordinal=device_ordinal, combiners=combiners, name=name)
    return _op