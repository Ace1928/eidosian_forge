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
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('quantization.fake_quant_with_min_max_vars_per_channel', v1=['quantization.fake_quant_with_min_max_vars_per_channel', 'fake_quant_with_min_max_vars_per_channel'])
@deprecated_endpoints('fake_quant_with_min_max_vars_per_channel')
def fake_quant_with_min_max_vars_per_channel(inputs: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max: _atypes.TensorFuzzingAnnotation[_atypes.Float32], num_bits: int=8, narrow_range: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    """Fake-quantize the 'inputs' tensor of type float via per-channel floats

  Fake-quantize the `inputs` tensor of type float per-channel and one of the
  shapes: `[d]`, `[b, d]` `[b, h, w, d]` via per-channel floats `min` and `max`
  of shape `[d]` to `outputs` tensor of same shape as `inputs`.

  Attributes

  *   `[min; max]` define the clamping range for the `inputs` data.
  *   `inputs` values are quantized into the quantization range (
  `[0; 2^num_bits - 1]` when `narrow_range` is false and `[1; 2^num_bits - 1]`
  when it is true) and then de-quantized and output as floats in `[min; max]`
  interval.
  *   `num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.

  Before quantization, `min` and `max` values are adjusted with the following
  logic.
  It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
  the behavior can be unexpected:

  *   If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
  *   If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
  *   If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
  `min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.

  This operation has a gradient and thus allows for training `min` and `max`
  values.

  Args:
    inputs: A `Tensor` of type `float32`.
    min: A `Tensor` of type `float32`.
    max: A `Tensor` of type `float32`.
    num_bits: An optional `int`. Defaults to `8`.
    narrow_range: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'FakeQuantWithMinMaxVarsPerChannel', name, inputs, min, max, 'num_bits', num_bits, 'narrow_range', narrow_range)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_fake_quant_with_min_max_vars_per_channel((inputs, min, max, num_bits, narrow_range, name), None)
            if _result is not NotImplemented:
                return _result
            return fake_quant_with_min_max_vars_per_channel_eager_fallback(inputs, min, max, num_bits=num_bits, narrow_range=narrow_range, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(fake_quant_with_min_max_vars_per_channel, (), dict(inputs=inputs, min=min, max=max, num_bits=num_bits, narrow_range=narrow_range, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_fake_quant_with_min_max_vars_per_channel((inputs, min, max, num_bits, narrow_range, name), None)
        if _result is not NotImplemented:
            return _result
    if num_bits is None:
        num_bits = 8
    num_bits = _execute.make_int(num_bits, 'num_bits')
    if narrow_range is None:
        narrow_range = False
    narrow_range = _execute.make_bool(narrow_range, 'narrow_range')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('FakeQuantWithMinMaxVarsPerChannel', inputs=inputs, min=min, max=max, num_bits=num_bits, narrow_range=narrow_range, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(fake_quant_with_min_max_vars_per_channel, (), dict(inputs=inputs, min=min, max=max, num_bits=num_bits, narrow_range=narrow_range, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('num_bits', _op._get_attr_int('num_bits'), 'narrow_range', _op._get_attr_bool('narrow_range'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('FakeQuantWithMinMaxVarsPerChannel', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result