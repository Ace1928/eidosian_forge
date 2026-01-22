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
def dilation2d(input: _atypes.TensorFuzzingAnnotation[TV_Dilation2D_T], filter: _atypes.TensorFuzzingAnnotation[TV_Dilation2D_T], strides, rates, padding: str, name=None) -> _atypes.TensorFuzzingAnnotation[TV_Dilation2D_T]:
    """Computes the grayscale dilation of 4-D `input` and 3-D `filter` tensors.

  The `input` tensor has shape `[batch, in_height, in_width, depth]` and the
  `filter` tensor has shape `[filter_height, filter_width, depth]`, i.e., each
  input channel is processed independently of the others with its own structuring
  function. The `output` tensor has shape
  `[batch, out_height, out_width, depth]`. The spatial dimensions of the output
  tensor depend on the `padding` algorithm. We currently only support the default
  "NHWC" `data_format`.

  In detail, the grayscale morphological 2-D dilation is the max-sum correlation
  (for consistency with `conv2d`, we use unmirrored filters):

      output[b, y, x, c] =
         max_{dy, dx} input[b,
                            strides[1] * y + rates[1] * dy,
                            strides[2] * x + rates[2] * dx,
                            c] +
                      filter[dy, dx, c]

  Max-pooling is a special case when the filter has size equal to the pooling
  kernel size and contains all zeros.

  Note on duality: The dilation of `input` by the `filter` is equal to the
  negation of the erosion of `-input` by the reflected `filter`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      4-D with shape `[batch, in_height, in_width, depth]`.
    filter: A `Tensor`. Must have the same type as `input`.
      3-D with shape `[filter_height, filter_width, depth]`.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the input
      tensor. Must be: `[1, stride_height, stride_width, 1]`.
    rates: A list of `ints` that has length `>= 4`.
      The input stride for atrous morphological dilation. Must be:
      `[1, rate_height, rate_width, 1]`.
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
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'Dilation2D', name, input, filter, 'strides', strides, 'rates', rates, 'padding', padding)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return dilation2d_eager_fallback(input, filter, strides=strides, rates=rates, padding=padding, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(strides, (list, tuple)):
        raise TypeError("Expected list for 'strides' argument to 'dilation2d' Op, not %r." % strides)
    strides = [_execute.make_int(_i, 'strides') for _i in strides]
    if not isinstance(rates, (list, tuple)):
        raise TypeError("Expected list for 'rates' argument to 'dilation2d' Op, not %r." % rates)
    rates = [_execute.make_int(_i, 'rates') for _i in rates]
    padding = _execute.make_str(padding, 'padding')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('Dilation2D', input=input, filter=filter, strides=strides, rates=rates, padding=padding, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'strides', _op.get_attr('strides'), 'rates', _op.get_attr('rates'), 'padding', _op.get_attr('padding'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('Dilation2D', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result