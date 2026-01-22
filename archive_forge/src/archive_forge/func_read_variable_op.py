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
def read_variable_op(resource: _atypes.TensorFuzzingAnnotation[_atypes.Resource], dtype: TV_ReadVariableOp_dtype, name=None) -> _atypes.TensorFuzzingAnnotation[TV_ReadVariableOp_dtype]:
    """Reads the value of a variable.

  The tensor returned by this operation is immutable.

  The value returned by this operation is guaranteed to be influenced by all the
  writes on which this operation depends directly or indirectly, and to not be
  influenced by any of the writes which depend directly or indirectly on this
  operation.

  Args:
    resource: A `Tensor` of type `resource`.
      handle to the resource in which to store the variable.
    dtype: A `tf.DType`. the dtype of the value.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ReadVariableOp', name, resource, 'dtype', dtype)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return read_variable_op_eager_fallback(resource, dtype=dtype, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    dtype = _execute.make_type(dtype, 'dtype')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ReadVariableOp', resource=resource, dtype=dtype, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtype', _op._get_attr_type('dtype'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ReadVariableOp', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result