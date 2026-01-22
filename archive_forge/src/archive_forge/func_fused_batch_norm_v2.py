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
def fused_batch_norm_v2(x: _atypes.TensorFuzzingAnnotation[TV_FusedBatchNormV2_T], scale: _atypes.TensorFuzzingAnnotation[TV_FusedBatchNormV2_U], offset: _atypes.TensorFuzzingAnnotation[TV_FusedBatchNormV2_U], mean: _atypes.TensorFuzzingAnnotation[TV_FusedBatchNormV2_U], variance: _atypes.TensorFuzzingAnnotation[TV_FusedBatchNormV2_U], epsilon: float=0.0001, exponential_avg_factor: float=1, data_format: str='NHWC', is_training: bool=True, name=None):
    """Batch normalization.

  Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      A 4D Tensor for input data.
    scale: A `Tensor`. Must be one of the following types: `float32`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    offset: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for offset, to shift to the normalized x.
    mean: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for population mean. Used for inference only;
      must be empty for training.
    variance: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for population variance. Used for inference only;
      must be empty for training.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    exponential_avg_factor: An optional `float`. Defaults to `1`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      The data format for x and y. Either "NHWC" (default) or "NCHW".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, batch_mean, batch_variance, reserve_space_1, reserve_space_2).

    y: A `Tensor`. Has the same type as `x`.
    batch_mean: A `Tensor`. Has the same type as `scale`.
    batch_variance: A `Tensor`. Has the same type as `scale`.
    reserve_space_1: A `Tensor`. Has the same type as `scale`.
    reserve_space_2: A `Tensor`. Has the same type as `scale`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'FusedBatchNormV2', name, x, scale, offset, mean, variance, 'epsilon', epsilon, 'exponential_avg_factor', exponential_avg_factor, 'data_format', data_format, 'is_training', is_training)
            _result = _FusedBatchNormV2Output._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return fused_batch_norm_v2_eager_fallback(x, scale, offset, mean, variance, epsilon=epsilon, exponential_avg_factor=exponential_avg_factor, data_format=data_format, is_training=is_training, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if epsilon is None:
        epsilon = 0.0001
    epsilon = _execute.make_float(epsilon, 'epsilon')
    if exponential_avg_factor is None:
        exponential_avg_factor = 1
    exponential_avg_factor = _execute.make_float(exponential_avg_factor, 'exponential_avg_factor')
    if data_format is None:
        data_format = 'NHWC'
    data_format = _execute.make_str(data_format, 'data_format')
    if is_training is None:
        is_training = True
    is_training = _execute.make_bool(is_training, 'is_training')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('FusedBatchNormV2', x=x, scale=scale, offset=offset, mean=mean, variance=variance, epsilon=epsilon, exponential_avg_factor=exponential_avg_factor, data_format=data_format, is_training=is_training, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'U', _op._get_attr_type('U'), 'epsilon', _op.get_attr('epsilon'), 'exponential_avg_factor', _op.get_attr('exponential_avg_factor'), 'data_format', _op.get_attr('data_format'), 'is_training', _op._get_attr_bool('is_training'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('FusedBatchNormV2', _inputs_flat, _attrs, _result)
    _result = _FusedBatchNormV2Output._make(_result)
    return _result