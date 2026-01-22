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
def batch_matrix_band_part(input: _atypes.TensorFuzzingAnnotation[TV_BatchMatrixBandPart_T], num_lower: _atypes.TensorFuzzingAnnotation[_atypes.Int64], num_upper: _atypes.TensorFuzzingAnnotation[_atypes.Int64], name=None) -> _atypes.TensorFuzzingAnnotation[TV_BatchMatrixBandPart_T]:
    """TODO: add doc.

  Args:
    input: A `Tensor`.
    num_lower: A `Tensor` of type `int64`.
    num_upper: A `Tensor` of type `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BatchMatrixBandPart', name, input, num_lower, num_upper)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return batch_matrix_band_part_eager_fallback(input, num_lower, num_upper, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BatchMatrixBandPart', input=input, num_lower=num_lower, num_upper=num_upper, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('BatchMatrixBandPart', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result