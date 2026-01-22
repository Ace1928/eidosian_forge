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
def fused_pad_conv2d(input: _atypes.TensorFuzzingAnnotation[TV_FusedPadConv2D_T], paddings: _atypes.TensorFuzzingAnnotation[_atypes.Int32], filter: _atypes.TensorFuzzingAnnotation[TV_FusedPadConv2D_T], mode: str, strides, padding: str, name=None) -> _atypes.TensorFuzzingAnnotation[TV_FusedPadConv2D_T]:
    """Performs a padding as a preprocess during a convolution.

  Similar to FusedResizeAndPadConv2d, this op allows for an optimized
  implementation where the spatial padding transformation stage is fused with the
  im2col lookup, but in this case without the bilinear filtering required for
  resizing. Fusing the padding prevents the need to write out the intermediate
  results as whole tensors, reducing memory pressure, and we can get some latency
  gains by merging the transformation calculations.
  The data_format attribute for Conv2D isn't supported by this op, and 'NHWC'
  order is used instead.
  Internally this op uses a single per-graph scratch buffer, which means that it
  will block if multiple versions are being run in parallel. This is because this
  operator is primarily an optimization to minimize memory usage.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape `[batch, in_height, in_width, in_channels]`.
    paddings: A `Tensor` of type `int32`.
      A two-column matrix specifying the padding sizes. The number of
      rows must be the same as the rank of `input`.
    filter: A `Tensor`. Must have the same type as `input`. 4-D with shape
      `[filter_height, filter_width, in_channels, out_channels]`.
    mode: A `string` from: `"REFLECT", "SYMMETRIC"`.
    strides: A list of `ints`.
      1-D of length 4.  The stride of the sliding window for each dimension
      of `input`. Must be in the same order as the dimension specified with format.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'FusedPadConv2D', name, input, paddings, filter, 'mode', mode, 'strides', strides, 'padding', padding)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return fused_pad_conv2d_eager_fallback(input, paddings, filter, mode=mode, strides=strides, padding=padding, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    mode = _execute.make_str(mode, 'mode')
    if not isinstance(strides, (list, tuple)):
        raise TypeError("Expected list for 'strides' argument to 'fused_pad_conv2d' Op, not %r." % strides)
    strides = [_execute.make_int(_i, 'strides') for _i in strides]
    padding = _execute.make_str(padding, 'padding')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('FusedPadConv2D', input=input, paddings=paddings, filter=filter, mode=mode, strides=strides, padding=padding, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'mode', _op.get_attr('mode'), 'strides', _op.get_attr('strides'), 'padding', _op.get_attr('padding'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('FusedPadConv2D', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result