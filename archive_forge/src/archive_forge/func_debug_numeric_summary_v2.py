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
def debug_numeric_summary_v2(input: _atypes.TensorFuzzingAnnotation[TV_DebugNumericSummaryV2_T], output_dtype: TV_DebugNumericSummaryV2_output_dtype=_dtypes.float32, tensor_debug_mode: int=-1, tensor_id: int=-1, name=None) -> _atypes.TensorFuzzingAnnotation[TV_DebugNumericSummaryV2_output_dtype]:
    """Debug Numeric Summary V2 Op.

  Computes a numeric summary of the input tensor. The shape of the output
  depends on the tensor_debug_mode attribute.
  This op is used internally by TensorFlow Debugger (tfdbg) v2.

  Args:
    input: A `Tensor`. Input tensor, to be summarized by the op.
    output_dtype: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
      Optional. The type of the output. Can be float32 or float64 (default: float32).
    tensor_debug_mode: An optional `int`. Defaults to `-1`.
      Tensor debug mode: the mode in which the input tensor is summarized
        by the op. See the TensorDebugMode enum in
        tensorflow/core/protobuf/debug_event.proto for details.

      Supported values:
        2 (CURT_HEALTH): Output a float32/64 tensor of shape [2]. The 1st
        element is the tensor_id, if provided, and -1 otherwise. The 2nd
        element is a bit which is set to 1 if the input tensor has an
        infinity or nan value, or zero otherwise.

        3 (CONCISE_HEALTH): Output a float32/64 tensor of shape [5]. The 1st
        element is the tensor_id, if provided, and -1 otherwise. The
        remaining four slots are the total number of elements, -infs,
        +infs, and nans in the input tensor respectively.

        4 (FULL_HEALTH): Output a float32/64 tensor of shape [11]. The 1st
        element is the tensor_id, if provided, and -1 otherwise. The 2nd
        element is the device_id, if provided, and -1 otherwise. The 3rd
        element holds the datatype value of the input tensor as according
        to the enumerated type in tensorflow/core/framework/types.proto.
        The remaining elements hold the total number of elements, -infs,
        +infs, nans, negative finite numbers, zeros, and positive finite
        numbers in the input tensor respectively.

        5 (SHAPE): Output a float32/64 tensor of shape [10]. The 1st
        element is the tensor_id, if provided, and -1 otherwise. The 2nd
        element holds the datatype value of the input tensor as according
        to the enumerated type in tensorflow/core/framework/types.proto.
        The 3rd element holds the rank of the tensor. The 4th element holds
        the number of elements within the tensor. Finally the remaining 6
        elements hold the shape of the tensor. If the rank of the tensor
        is lower than 6, the shape is right padded with zeros. If the rank
        is greater than 6, the head of the shape is truncated.

        6 (FULL_NUMERICS): Output a float32/64 tensor of shape [22]. The 1st
        element is the tensor_id, if provided, and -1 otherwise. The 2nd
        element is the device_id, if provided, and -1 otherwise. The 3rd
        element holds the datatype value of the input tensor as according
        to the enumerated type in tensorflow/core/framework/types.proto.
        The 4th element holds the rank of the tensor. The 5th to 11th
        elements hold the shape of the tensor. If the rank of the tensor
        is lower than 6, the shape is right padded with zeros. If the rank
        is greater than 6, the head of the shape is truncated. The 12th to
        18th elements hold the number of elements, -infs, +infs, nans,
        denormal floats, negative finite numbers, zeros, and positive
        finite numbers in the input tensor respectively. The final four
        elements hold the min value, max value, mean, and variance of the
        input tensor.

        8 (REDUCE_INF_NAN_THREE_SLOTS): Output a float32/64 tensor of shape
        [3]. The 1st element is -inf if any elements of the input tensor
        is -inf, or zero otherwise. The 2nd element is +inf if any elements
        of the input tensor is +inf, or zero otherwise.  The 3rd element is
        nan if any element of the input tensor is nan, or zero otherwise.
    tensor_id: An optional `int`. Defaults to `-1`.
      Optional. An integer identifier for the tensor being summarized by this op.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `output_dtype`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DebugNumericSummaryV2', name, input, 'output_dtype', output_dtype, 'tensor_debug_mode', tensor_debug_mode, 'tensor_id', tensor_id)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return debug_numeric_summary_v2_eager_fallback(input, output_dtype=output_dtype, tensor_debug_mode=tensor_debug_mode, tensor_id=tensor_id, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if output_dtype is None:
        output_dtype = _dtypes.float32
    output_dtype = _execute.make_type(output_dtype, 'output_dtype')
    if tensor_debug_mode is None:
        tensor_debug_mode = -1
    tensor_debug_mode = _execute.make_int(tensor_debug_mode, 'tensor_debug_mode')
    if tensor_id is None:
        tensor_id = -1
    tensor_id = _execute.make_int(tensor_id, 'tensor_id')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('DebugNumericSummaryV2', input=input, output_dtype=output_dtype, tensor_debug_mode=tensor_debug_mode, tensor_id=tensor_id, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('output_dtype', _op._get_attr_type('output_dtype'), 'T', _op._get_attr_type('T'), 'tensor_debug_mode', _op._get_attr_int('tensor_debug_mode'), 'tensor_id', _op._get_attr_int('tensor_id'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('DebugNumericSummaryV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result