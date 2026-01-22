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
def destroy_temporary_variable(ref: _atypes.TensorFuzzingAnnotation[TV_DestroyTemporaryVariable_T], var_name: str, name=None) -> _atypes.TensorFuzzingAnnotation[TV_DestroyTemporaryVariable_T]:
    """Destroys the temporary variable and returns its final value.

  Sets output to the value of the Tensor pointed to by 'ref', then destroys
  the temporary variable called 'var_name'.
  All other uses of 'ref' *must* have executed before this op.
  This is typically achieved by chaining the ref through each assign op, or by
  using control dependencies.

  Outputs the final value of the tensor pointed to by 'ref'.

  Args:
    ref: A mutable `Tensor`. A reference to the temporary variable tensor.
    var_name: A `string`.
      Name of the temporary variable, usually the name of the matching
      'TemporaryVariable' op.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `ref`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("destroy_temporary_variable op does not support eager execution. Arg 'ref' is a ref.")
    var_name = _execute.make_str(var_name, 'var_name')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('DestroyTemporaryVariable', ref=ref, var_name=var_name, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'var_name', _op.get_attr('var_name'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('DestroyTemporaryVariable', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result