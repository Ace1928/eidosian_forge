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
def assign_add_variable_op(resource: _atypes.TensorFuzzingAnnotation[_atypes.Resource], value: _atypes.TensorFuzzingAnnotation[TV_AssignAddVariableOp_dtype], name=None):
    """Adds a value to the current value of a variable.

  Any ReadVariableOp with a control dependency on this op is guaranteed to
  see the incremented value or a subsequent newer one.

  Args:
    resource: A `Tensor` of type `resource`.
      handle to the resource in which to store the variable.
    value: A `Tensor`. the value by which the variable will be incremented.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'AssignAddVariableOp', name, resource, value)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return assign_add_variable_op_eager_fallback(resource, value, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('AssignAddVariableOp', resource=resource, value=value, name=name)
    return _op