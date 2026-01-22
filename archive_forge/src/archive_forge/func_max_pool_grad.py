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
def max_pool_grad(orig_input: _atypes.TensorFuzzingAnnotation[TV_MaxPoolGrad_T], orig_output: _atypes.TensorFuzzingAnnotation[TV_MaxPoolGrad_T], grad: _atypes.TensorFuzzingAnnotation[TV_MaxPoolGrad_T], ksize, strides, padding: str, explicit_paddings=[], data_format: str='NHWC', name=None) -> _atypes.TensorFuzzingAnnotation[TV_MaxPoolGrad_T]:
    """Computes gradients of the maxpooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must have the same type as `orig_input`.
      4-D.  Gradients w.r.t. the output of `max_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID", "EXPLICIT"`.
      The type of padding algorithm to use.
    explicit_paddings: An optional list of `ints`. Defaults to `[]`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'MaxPoolGrad', name, orig_input, orig_output, grad, 'ksize', ksize, 'strides', strides, 'padding', padding, 'explicit_paddings', explicit_paddings, 'data_format', data_format)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return max_pool_grad_eager_fallback(orig_input, orig_output, grad, ksize=ksize, strides=strides, padding=padding, explicit_paddings=explicit_paddings, data_format=data_format, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(ksize, (list, tuple)):
        raise TypeError("Expected list for 'ksize' argument to 'max_pool_grad' Op, not %r." % ksize)
    ksize = [_execute.make_int(_i, 'ksize') for _i in ksize]
    if not isinstance(strides, (list, tuple)):
        raise TypeError("Expected list for 'strides' argument to 'max_pool_grad' Op, not %r." % strides)
    strides = [_execute.make_int(_i, 'strides') for _i in strides]
    padding = _execute.make_str(padding, 'padding')
    if explicit_paddings is None:
        explicit_paddings = []
    if not isinstance(explicit_paddings, (list, tuple)):
        raise TypeError("Expected list for 'explicit_paddings' argument to 'max_pool_grad' Op, not %r." % explicit_paddings)
    explicit_paddings = [_execute.make_int(_i, 'explicit_paddings') for _i in explicit_paddings]
    if data_format is None:
        data_format = 'NHWC'
    data_format = _execute.make_str(data_format, 'data_format')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('MaxPoolGrad', orig_input=orig_input, orig_output=orig_output, grad=grad, ksize=ksize, strides=strides, padding=padding, explicit_paddings=explicit_paddings, data_format=data_format, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('ksize', _op.get_attr('ksize'), 'strides', _op.get_attr('strides'), 'padding', _op.get_attr('padding'), 'explicit_paddings', _op.get_attr('explicit_paddings'), 'data_format', _op.get_attr('data_format'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('MaxPoolGrad', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result