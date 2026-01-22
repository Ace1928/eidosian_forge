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
def requantization_range_per_channel(input: _atypes.TensorFuzzingAnnotation[TV_RequantizationRangePerChannel_T], input_min: _atypes.TensorFuzzingAnnotation[_atypes.Float32], input_max: _atypes.TensorFuzzingAnnotation[_atypes.Float32], clip_value_max: float, name=None):
    """Computes requantization range per channel.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    input_min: A `Tensor` of type `float32`.
      The minimum value of the input tensor
    input_max: A `Tensor` of type `float32`.
      The maximum value of the input tensor.
    clip_value_max: A `float`.
      The maximum value of the output that needs to be clipped.
      Example: set this to 6 for Relu6.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_min, output_max).

    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RequantizationRangePerChannel', name, input, input_min, input_max, 'clip_value_max', clip_value_max)
            _result = _RequantizationRangePerChannelOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return requantization_range_per_channel_eager_fallback(input, input_min, input_max, clip_value_max=clip_value_max, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    clip_value_max = _execute.make_float(clip_value_max, 'clip_value_max')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RequantizationRangePerChannel', input=input, input_min=input_min, input_max=input_max, clip_value_max=clip_value_max, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'clip_value_max', _op.get_attr('clip_value_max'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RequantizationRangePerChannel', _inputs_flat, _attrs, _result)
    _result = _RequantizationRangePerChannelOutput._make(_result)
    return _result